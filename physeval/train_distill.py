"""Lightweight distillation fine-tuning harness for PhysEval-Agent.

Fine-tunes a causal LM on PhysEval supervision using TRL:

* **SFT stage** (:class:`trl.SFTTrainer`) -- trains on *successful* PRM steps
  (reward ``1.0``) from ``data/prm_steps.jsonl``, formatted as system/user/
  assistant conversations and rendered through the tokenizer's chat template;
  supports efficient context packing.
* **DPO stage** (:class:`trl.DPOTrainer`) -- trains on preference pairs from
  ``data/dpo_pairs.jsonl`` (unverified code vs. oracle-corrected patch),
  conditioned on the exact diagnostic repair prompt.

Supports parameter-efficient tuning via PEFT: plain **LoRA** or **QLoRA**
(NF4 4-bit quantization through ``bitsandbytes``), plus full fine-tuning.

Custom :class:`PhysicsTokenMetrics` callbacks attach physics-vocabulary
statistics (term density per 1k tokens, lexicon coverage) to every evaluation
cycle, so Weights & Biases / TensorBoard show whether the distilled model is
absorbing domain vocabulary alongside validation loss.

Heavy dependencies (torch, transformers, trl, peft, bitsandbytes, datasets,
wandb) are imported lazily; install with::

    pip install 'physeval-agent[train]'

Example:
    python -m physeval.train_distill --stage sft \\
        --base-model Qwen/Qwen2.5-Coder-7B-Instruct --method qlora \\
        --report-to wandb --output-dir runs/sft-qlora
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

LOGGER = logging.getLogger("physeval.train")

#: Domain vocabulary tracked as a training health signal.
PHYSICS_LEXICON: Tuple[str, ...] = (
    "cfl", "courant", "kirchhoff", "nodal", "imbalance", "soc",
    "curtailment", "thermal limit", "p_max_pu", "p_nom", "dispatch",
    "storage unit", "ramp limit", "tracer", "advection", "diffusion",
    "flux-form", "conservation", "periodic boundary", "isotherm",
    "langmuir", "adsorption", "desorption", "steady state", "van't hoff",
    "arrhenius", "netcdf", "xarray", "pypsa", "power flow",
)

DEFAULT_SYSTEM_PROMPT = (
    "You are PhysEval-Agent, an expert scientific-computing engineer. You "
    "write complete, deterministic Python simulations that satisfy physical "
    "conservation laws exactly."
)


# --------------------------------------------------------------------------- #
# Data loading and formatting                                                 #
# --------------------------------------------------------------------------- #

def iter_jsonl(path: str | Path) -> Iterable[Dict[str, Any]]:
    """Yield parsed JSON objects from *path*, skipping blank/malformed lines."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Training data not found: {file_path}")
    with file_path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                LOGGER.warning("Skipping malformed line %d in %s: %s", lineno, file_path, exc)
                continue
            if isinstance(obj, dict):
                yield obj


def load_sft_conversations(
    prm_path: str | Path,
    *,
    max_prompts: Optional[int] = None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> Tuple[List[List[Dict[str, str]]], Dict[str, int]]:
    """Build chat conversations from successful PRM steps.

    A step qualifies when its binary process reward is ``1.0`` (sandbox run
    succeeded and all physical invariants held) and it carries non-empty
    solution code.

    Returns:
        ``(conversations, filter_stats)`` where each conversation is a list of
        ``{"role", "content"}`` messages and the stats describe dropped rows.
    """
    convos: List[List[Dict[str, str]]] = []
    kept = skipped_reward = skipped_empty = skipped_long = 0
    max_chars = 24_000
    for row in iter_jsonl(prm_path):
        if max_prompts is not None and len(convos) >= max_prompts:
            break
        if float(row.get("reward") or 0.0) != 1.0:
            skipped_reward += 1
            continue
        code = row.get("code")
        prompt = row.get("prompt")
        if not isinstance(code, str) or not code.strip() or not isinstance(prompt, str) \
                or not prompt.strip():
            skipped_empty += 1
            continue
        if len(prompt) > max_chars:
            skipped_long += 1
            continue
        convos.append([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": code},
        ])
        kept += 1
    stats = {
        "kept": kept,
        "skipped_reward_zero": skipped_reward,
        "skipped_empty": skipped_empty,
        "skipped_overlong": skipped_long,
    }
    if not convos:
        raise ValueError(
            f"No qualifying SFT examples in {prm_path}. Need steps with reward==1.0 "
            "and embedded code; run rollouts first (physeval.run_rollouts) and "
            "export with physeval.export_dataset."
        )
    return convos, stats


def load_dpo_preferences(
    dpo_path: str | Path,
    *,
    max_pairs: Optional[int] = None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> Tuple[List[Dict[str, List[Dict[str, str]]]], Dict[str, int]]:
    """Build preference triplets from ``dpo_pairs.jsonl``.

    Each record has ``prompt`` (system+user repair context), ``chosen``
    (oracle-corrected patch) and ``rejected`` (failed unverified code).
    """
    triples: List[Dict[str, List[Dict[str, str]]]] = []
    kept = skipped_short = 0
    for row in iter_jsonl(dpo_path):
        if max_pairs is not None and len(triples) >= max_pairs:
            break
        prompt, chosen, rejected = row.get("prompt"), row.get("chosen"), row.get("rejected")
        if not all(isinstance(x, str) and x.strip() for x in (prompt, chosen, rejected)):
            skipped_short += 1
            continue
        if chosen.strip() == rejected.strip():
            skipped_short += 1
            continue
        triples.append({
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "chosen": [{"role": "assistant", "content": chosen}],
            "rejected": [{"role": "assistant", "content": rejected}],
        })
        kept += 1
    stats = {"kept": kept, "skipped_invalid_or_identical": skipped_short}
    if not triples:
        raise ValueError(
            f"No usable DPO pairs in {dpo_path}. Export pairs with "
            "physeval.export_dataset after running self-correction rollouts."
        )
    return triples, stats


# --------------------------------------------------------------------------- #
# Physics-vocabulary metrics                                                  #
# --------------------------------------------------------------------------- #

def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def physics_token_stats(
    texts: Sequence[str],
    *,
    lexicon: Sequence[str] = PHYSICS_LEXICON,
    tokenizer: Any = None,
) -> Dict[str, float]:
    """Compute physics-vocabulary statistics over *texts*.

    Metrics:
        ``physics_terms_per_1k_tokens`` -- normalized term density; uses the
        real tokenizer when supplied, else a whitespace approximation.
        ``lexicon_coverage`` -- fraction of lexicon terms observed at least once.
    """
    corpus = "\n".join(_norm(t or "") for t in texts)
    total_tokens = 0
    if tokenizer is not None:
        try:
            encoded = tokenizer(corpus, add_special_tokens=False)
            total_tokens = max(1, len(encoded.get("input_ids", [])))
        except Exception:
            total_tokens = 0
    if total_tokens == 0:
        total_tokens = max(1, len(corpus.split()))

    hits = 0
    seen = 0
    for term in lexicon:
        count = corpus.count(_norm(term))
        hits += count
        seen += 1 if count > 0 else 0
    return {
        "physics_terms_per_1k_tokens": round(1000.0 * hits / total_tokens, 4),
        "lexicon_coverage": round(seen / max(1, len(tuple(lexicon))), 4),
    }


try:  # pragma: no cover - exercised only with the [train] extra installed
    from transformers.trainer_callback import TrainerCallback

    class PhysicsTokenMetrics(TrainerCallback):  # type: ignore[misc,name-defined]
        """Attach physics-vocabulary metrics to every evaluation cycle."""

        def __init__(self, eval_texts: Sequence[str]) -> None:
            self._stats = physics_token_stats(list(eval_texts))

        def on_evaluate(self, args: Any, state: Any, metrics: Dict[str, Any],
                        **kwargs: Any) -> None:
            metrics["physics_terms_per_1k_tokens"] = self._stats[
                "physics_terms_per_1k_tokens"
            ]
            metrics["lexicon_coverage"] = self._stats["lexicon_coverage"]

except ImportError:  # pragma: no cover - keeps module importable without torch

    class PhysicsTokenMetrics:  # type: ignore[no-redef]
        """Placeholder raising on use; install the [train] extra for real one."""

        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise ImportError(
                "PhysicsTokenMetrics requires 'transformers'; "
                "install with: pip install 'physeval-agent[train]'"
            )


# --------------------------------------------------------------------------- #
# Harness                                                                     #
# --------------------------------------------------------------------------- #

def _require(extra_hint: str = "train") -> Callable[[str], None]:
    def fail(module: str) -> None:
        raise ImportError(
            f"Missing '{module}'; install training extras with: "
            f"pip install 'physeval-agent[{extra_hint}]'"
        )
    return fail


def _split_conversations(
    convos: List[List[Dict[str, str]]],
    eval_fraction: float,
    seed: int,
) -> Tuple[List[List[Dict[str, str]]], List[List[Dict[str, str]]]]:
    """Deterministic hold-out split without requiring the datasets library."""
    if not 0.0 <= eval_fraction < 1.0:
        raise ValueError("eval_fraction must be in [0, 1).")
    ordered = sorted(range(len(convos)))  # already deterministic order
    n_eval = math.floor(len(convos) * eval_fraction)
    eval_idx = set(ordered[:n_eval])
    train = [c for i, c in enumerate(convos) if i not in eval_idx]
    eval_part = [c for i, c in enumerate(convos) if i in eval_idx]
    del seed  # ordering is input-deterministic; kept for API stability
    return train, eval_part


class DistillationHarness:
    """End-to-end SFT/DPO fine-tuning with PEFT (LoRA/QLoRA) support."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.fail = _require()

    # --------------------------- common pieces ------------------------ #

    def _load_tokenizer_and_model(self) -> Tuple[Any, Any]:
        fail = self.fail
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            fail("transformers/torch")
            raise  # unreachable; satisfies type checkers

        tok = AutoTokenizer.from_pretrained(self.args.base_model, trust_remote_code=False)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        quantization_config = None
        if self.args.method == "qlora":
            try:
                from transformers import BitsAndBytesConfig
            except ImportError:
                fail("bitsandbytes")
                raise
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

        dtype = getattr(torch, self.args.torch_dtype, None) or torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            self.args.base_model,
            quantization_config=quantization_config,
            torch_dtype=dtype,
            device_map=self.args.device_map,
            trust_remote_code=False,
            attn_implementation=self.args.attn_implementation,
        )
        model.config.use_cache = False
        return tok, model

    def _maybe_attach_lora(self, model: Any) -> Any:
        if self.args.method == "full":
            return model
        try:
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        except ImportError:
            self.fail("peft")
            raise
        if self.args.method == "qlora":
            model = prepare_model_for_kbit_training(model)
        targets = [t.strip() for t in self.args.lora_target_modules.split(",") if t.strip()]
        lora_cfg = LoraConfig(
            r=self.args.lora_r,
            lora_alpha=self.args.lora_alpha,
            lora_dropout=self.args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=targets or None,
        )
        return get_peft_model(model, lora_cfg)

    @staticmethod
    def _training_kwargs(args: argparse.Namespace, *, eval_rows: int) -> Dict[str, Any]:
        """Build TrainingArguments-compatible kwargs across versions."""
        kwargs: Dict[str, Any] = dict(
            output_dir=args.output_dir,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            num_train_epochs=args.epochs,
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,
            logging_steps=args.logging_steps,
            save_strategy="epoch",
            bf16=args.bf16,
            gradient_checkpointing=True,
            report_to=[] if args.report_to == "none" else [args.report_to],
            seed=args.seed,
        )
        if eval_rows > 0:
            kwargs["per_device_eval_batch_size"] = args.batch_size
            # transformers renamed evaluation_strategy -> eval_strategy (4.46).
            kwargs["eval_strategy"] = "epoch"
        return kwargs

    @staticmethod
    def _patch_strategy_kwarg(kwargs: Dict[str, Any]) -> None:
        """Rename eval_strategy for older transformers releases."""
        try:
            import transformers

            major_minor = tuple(int(x) for x in transformers.__version__.split(".")[:2])
            if major_minor < (4, 46):
                kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")
        except Exception:
            pass

    # ------------------------------ stages ----------------------------- #

    def run_sft(self) -> Path:
        """Run supervised fine-tuning on successful rollout steps."""
        # Validate inputs before importing heavy dependencies so users get
        # actionable data errors rather than environment errors.
        convos, filter_stats = load_sft_conversations(
            self.args.sft_data, max_prompts=self.args.max_examples
        )
        LOGGER.info("SFT data: %s", filter_stats)
        train_convos, eval_convos = _split_conversations(
            convos, self.args.eval_fraction, self.args.seed
        )

        try:
            from datasets import Dataset
            from trl import SFTConfig, SFTTrainer
        except ImportError:
            self.fail("trl/datasets")
            raise

        tokenizer, model = self._load_tokenizer_and_model()
        model = self._maybe_attach_lora(model)

        def render(messages: List[Dict[str, str]]) -> str:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )

        train_rows = [{"text": render(c)} for c in train_convos]
        eval_rows = [{"text": render(c)} for c in eval_convos]
        train_ds = Dataset.from_list(train_rows)
        eval_ds = Dataset.from_list(eval_rows) if eval_rows else None

        tkw = self._training_kwargs(self.args, eval_rows=len(eval_rows))
        self._patch_strategy_kwarg(tkw)
        config = SFTConfig(
            packing=self.args.packing,
            max_length=self.args.max_len,
            dataset_text_field="text",
            **tkw,
        )
        callbacks = [PhysicsTokenMetrics([render(c) for c in eval_convos])] if eval_convos else []

        trainer = SFTTrainer(
            model=model,
            args=config,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            processing_class=tokenizer,
            callbacks=callbacks,
        )
        trainer.train()
        out_dir = Path(self.args.output_dir)
        trainer.save_model(str(out_dir))
        tokenizer.save_pretrained(str(out_dir))
        if self.args.merge_adapter and self.args.method != "full":
            merged = trainer.model.merge_and_unload()  # type: ignore[attr-defined]
            merged_dir = out_dir / "merged"
            merged.save_pretrained(str(merged_dir))
            tokenizer.save_pretrained(str(merged_dir))
            LOGGER.info("Merged adapter saved to %s", merged_dir)
        return out_dir

    def run_dpo(self) -> Path:
        """Run direct preference optimization on self-correction pairs."""
        # Validate inputs before importing heavy dependencies (see run_sft).
        triples, filter_stats = load_dpo_preferences(
            self.args.dpo_data, max_pairs=self.args.max_examples
        )
        LOGGER.info("DPO data: %s", filter_stats)
        train_triples, eval_triples = _split_conversations(
            triples, self.args.eval_fraction, self.args.seed
        )

        try:
            from datasets import Dataset
            from trl import DPOConfig, DPOTrainer
        except ImportError:
            self.fail("trl/datasets")
            raise

        tokenizer, model = self._load_tokenizer_and_model()
        model = self._maybe_attach_lora(model)

        eos = tokenizer.eos_token or ""

        def render_prompt(messages: List[Dict[str, str]]) -> str:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

        def completion_text(side: List[Dict[str, str]]) -> str:
            content = side[0]["content"]
            return f"{content}{eos}"

        def to_row(t: Dict[str, List[Dict[str, str]]]) -> Dict[str, str]:
            return {
                "prompt": render_prompt(t["prompt"]),
                "chosen": completion_text(t["chosen"]),
                "rejected": completion_text(t["rejected"]),
            }

        train_ds = Dataset.from_list([to_row(t) for t in train_triples])
        eval_ds = Dataset.from_list([to_row(t) for t in eval_triples]) if eval_triples else None

        tkw = self._training_kwargs(self.args, eval_rows=len(eval_triples))
        self._patch_strategy_kwarg(tkw)
        config = DPOConfig(
            beta=self.args.dpo_beta,
            max_prompt_length=self.args.max_len // 2,
            max_length=self.args.max_len,
            **tkw,
        )
        callbacks = [
            PhysicsTokenMetrics([
                render_prompt(t["prompt"]) + completion_text(t["chosen"])
                for t in eval_triples
            ])
        ] if eval_triples else []

        trainer = DPOTrainer(
            model=model,
            args=config,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            processing_class=tokenizer,
            callbacks=callbacks,
        )
        trainer.train()
        out_dir = Path(self.args.output_dir)
        trainer.save_model(str(out_dir))
        tokenizer.save_pretrained(str(out_dir))
        return out_dir

    def run(self) -> Path:
        """Dispatch to the configured training stage."""
        if self.args.stage == "sft":
            return self.run_sft()
        return self.run_dpo()


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #

def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="physeval-train-distill",
        description="Distill PhysEval-Agent behaviors into a code LLM (SFT/DPO + LoRA/QLoRA).",
    )
    parser.add_argument("--stage", required=True, choices=["sft", "dpo"])
    parser.add_argument("--sft-data", default="data/prm_steps.jsonl")
    parser.add_argument("--dpo-data", default="data/dpo_pairs.jsonl")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--method", default="lora", choices=["lora", "qlora", "full"])
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--eval-fraction", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-len", type=int, default=4096)
    parser.add_argument("--packing", action="store_true",
                        help="Enable sequence packing for the SFT stage.")
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", default=(
        "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"))
    parser.add_argument("--dpo-beta", type=float, default=0.1)
    parser.add_argument("--report-to", default="tensorboard",
                        choices=["wandb", "tensorboard", "none"])
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--no-bf16", dest="bf16", action="store_false")
    parser.add_argument("--torch-dtype", default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--attn-implementation", default="sdpa",
                        choices=["sdpa", "eager", "flash_attention_2"])
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--merge-adapter", action="store_true",
                        help="Merge LoRA weights into the base model after training.")
    args = parser.parse_args(argv)
    if args.output_dir is None:
        args.output_dir = f"runs/{args.stage}-{args.method}"
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point; returns a process exit code."""
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s :: %(message)s")
    harness = DistillationHarness(args)
    try:
        out_dir = harness.run()
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except ImportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    print(f"training artifacts written to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
