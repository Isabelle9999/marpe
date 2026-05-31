# single-agent probe training: train 3 probes on saved activations from gen.py
# probe types:
#   linear   -- linear-probe, sklearn logistic regression, eval with mean aggregation
#   attention -- attention-probe (causal self-attention), train/eval with last-position aggregation
#   max-pool  -- linear-probe architecture, train/eval with plain max-over-positions aggregation
#
# aggregator note for max-pool:
#   uses max_aggregator() = SequenceAggregator(method="max")
#   = max over sequence positions after summing heads (nhead=1, so sum is identity)
#   NOT multimax_aggregator() = SequenceAggregator(method="multimax")
#   = max per head then sum over heads (GDM paper eq 9, fancier multi-head variant)
#
# memory layout: acts split into per-layer files by split_acts.py (acts_layer_{i:02d}.pt)
# loop order: layer outer, probe_type inner -> 28 loads, 84 train_detector calls total
# results.json written incrementally after each (probe, layer) for crash recovery

import json
from pathlib import Path

import torch
from datasets import Dataset as HFDataset
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer
import numpy as np

from obfuscation_atlas.config import BaseTrainConfig
from obfuscation_atlas.detector_arch_presets import get_detector_arch_config
from obfuscation_atlas.detectors.dataset import ActivationDataset
from obfuscation_atlas.detectors.probe_archs import last_aggregator, max_aggregator, mean_aggregator
from obfuscation_atlas.detectors.train import get_detector_scores, train_detector

MODEL = "meta-llama/Llama-3.2-3B-Instruct"
ART = Path(__file__).parent / "artifacts" / "probes"
N_LAYERS = 28
PASS_AUROC = 0.85

# probe configurations: arch preset, sklearn flag, train aggregator, eval aggregator
# attention probe requires train_sequence_aggregator != None (atlas assertion)
# last_aggregator: takes logit at last valid position; correct for causal attention probe
PROBE_CFGS = {
    "linear": {
        "arch": get_detector_arch_config("linear-probe"),
        "use_sklearn": True,
        "train_agg": None,
        "eval_agg": mean_aggregator(),
    },
    "attention": {
        "arch": get_detector_arch_config("attention-probe"),
        "use_sklearn": False,
        "train_agg": last_aggregator(),
        "eval_agg": last_aggregator(),
    },
    "max-pool": {
        "arch": get_detector_arch_config("linear-probe"),
        "use_sklearn": False,
        "train_agg": max_aggregator(),
        "eval_agg": max_aggregator(),
    },
}


def build_act_ds_single(tok, layer_acts, layer_idx, tokens, masks, labels, idx):
    # fancy indexing creates copies independent of layer_acts so caller can del it
    sub_acts = {str(layer_idx): layer_acts[idx]}
    sub_tokens = tokens[idx]
    sub_pm = masks["prompt"][idx]
    sub_cm = masks["completion"][idx]
    sub_labels = labels[idx]
    return ActivationDataset(
        tokenizer=tok,
        activations=sub_acts,
        tokens_and_masks=(sub_tokens, sub_pm, sub_cm),
        labels=sub_labels,
        obfuscate_over="generation",
        preserve_seq_len=True,
        shuffle=False,
    )


def auroc_per_layer(scores_map, labels):
    if len(np.unique(labels)) < 2:
        return {}
    return {int(k): float(roc_auc_score(labels, v)) for k, v in scores_map.items()}


def main():
    for probe_name in PROBE_CFGS:
        (ART / "probes" / probe_name).mkdir(parents=True, exist_ok=True)

    print("loading tokenizer")
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print("loading metadata")
    labels = torch.load(ART / "labels_gemini.pt", map_location="cpu")
    tokens = torch.load(ART / "tokens.pt", map_location="cpu")
    masks = torch.load(ART / "masks.pt", map_location="cpu")
    text_pairs = json.loads((ART / "text_pairs.json").read_text())

    keep = labels != -1
    labels = labels[keep]
    tokens = tokens[keep]
    masks = {"prompt": masks["prompt"][keep], "completion": masks["completion"][keep]}
    text_pairs = [p for p, k in zip(text_pairs, keep.tolist()) if k]

    n = len(labels)
    n_pos = int(labels.sum().item())
    print(f"n={n}, pos={n_pos}, neg={n - n_pos}")

    # 80/20 split, seed 42
    rng = torch.Generator().manual_seed(42)
    idx = torch.randperm(n, generator=rng)
    n_train = int(0.8 * n)
    train_idx = idx[:n_train]
    test_idx = idx[n_train:]
    print(f"train={len(train_idx)}, test={len(test_idx)}")

    # hfdataset wrappers: required positional arg for train_detector api;
    # never accessed when train_feature_dataset is provided (prepare_dataset takes else branch)
    ti = train_idx.tolist()
    hf_pos = HFDataset.from_list([text_pairs[i] for i in ti if labels[i].item() == 1])
    hf_neg = HFDataset.from_list([text_pairs[i] for i in ti if labels[i].item() == 0])
    hf_train = (hf_pos, hf_neg)

    # published defaults from BaseTrainConfig; max_steps=3000 used as max_iter for sklearn
    train_cfg = BaseTrainConfig(device="cuda")

    all_results = []

    for layer in range(N_LAYERS):
        print(f"\nlayer {layer}")
        layer_acts = torch.load(ART / f"acts_layer_{layer:02d}.pt", map_location="cpu")
        layer_acts = layer_acts[keep]
        train_ds = build_act_ds_single(tok, layer_acts, layer, tokens, masks, labels, train_idx)
        test_ds = build_act_ds_single(tok, layer_acts, layer, tokens, masks, labels, test_idx)
        del layer_acts

        for probe_name, cfg in PROBE_CFGS.items():
            probe_dir = ART / "probes" / probe_name

            # model=None is safe here: train_feature_dataset is provided, so prepare_dataset
            # takes the else branch and never accesses model.config or model.device.
            # layers must be explicit to skip the get_num_hidden_layers(model.config) call.
            detector, _, _ = train_detector(
                model=None,
                tokenizer=tok,
                train_dataset=hf_train,
                train_cfg=train_cfg,
                layers=[layer],
                obfuscate_over="generation",
                detector_arch_config=cfg["arch"],
                train_sequence_aggregator=cfg["train_agg"],
                train_feature_dataset=train_ds,
                use_sklearn=cfg["use_sklearn"],
            )

            torch.save(detector[layer].state_dict(), probe_dir / f"layer_{layer:02d}.pt")

            # reset to preserve_seq_len=True for sequence-level scoring;
            # train_detector may have set preserve_seq_len=False (e.g. for sklearn token-level training)
            train_ds.set_obfuscation("generation", preserve_seq_len=True, balance_dataset=False)
            test_ds.set_obfuscation("generation", preserve_seq_len=True, balance_dataset=False)

            tr_scores, tr_labels, _ = get_detector_scores(
                detector, train_ds, eval_sequence_aggregator=cfg["eval_agg"]
            )
            te_scores, te_labels, _ = get_detector_scores(
                detector, test_ds, eval_sequence_aggregator=cfg["eval_agg"]
            )

            tr_auroc = auroc_per_layer(tr_scores, tr_labels)
            te_auroc = auroc_per_layer(te_scores, te_labels)

            row = {
                "probe_type": probe_name,
                "layer": layer,
                "train_auroc": round(tr_auroc.get(layer, -1.0), 4),
                "test_auroc": round(te_auroc.get(layer, -1.0), 4),
            }
            all_results.append(row)
            # incremental write for crash recovery
            (ART / "results_gemini.json").write_text(json.dumps(all_results, indent=2), encoding="utf-8")
            print(f"  {probe_name}: train {row['train_auroc']:.3f}  test {row['test_auroc']:.3f}")

            del detector
            torch.cuda.empty_cache()

        del train_ds, test_ds

    lines = [
        "# single-agent probe results (gemini labels)",
        "",
        f"model: {MODEL}",
        f"n_labeled: {n} ({n_pos} conceals, {n - n_pos} discloses-honestly)",
        f"split: {n_train} train / {n - n_train} test (80/20, seed 42)",
        f"pass criterion: test auroc >= {PASS_AUROC}",
        "",
    ]

    overall_pass = True
    for probe_name in PROBE_CFGS:
        rows = [r for r in all_results if r["probe_type"] == probe_name]
        if not rows:
            lines.append(f"## {probe_name}: no results")
            overall_pass = False
            continue
        best = max(rows, key=lambda r: r["test_auroc"])
        passed = best["test_auroc"] >= PASS_AUROC
        if not passed:
            overall_pass = False
        tag = "pass" if passed else "fail"
        lines.append(
            f"## {probe_name}: best layer {best['layer']}  "
            f"test auroc {best['test_auroc']:.4f}  [{tag}]"
        )
        lines.append(f"   train auroc at best layer: {best['train_auroc']:.4f}")
        lines.append("")

    lines.append(f"overall: {'PASS' if overall_pass else 'FAIL'}")
    (ART / "summary_gemini.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nresults written to {ART}")


if __name__ == "__main__":
    main()
