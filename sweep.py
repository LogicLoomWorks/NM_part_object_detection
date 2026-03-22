"""sweep.py — generate all (model × aug_preset × components) combinations.

Usage:
    python sweep.py --dry-run          # print full table and exit
    python sweep.py --sweep            # run every combination (20 epochs each)
    python sweep.py --sweep --n 2      # run only the first 2 combinations
    python sweep.py --sweep --epochs 1 # override epoch count (useful for testing)
    from sweep import generate_combinations, run_combination, sweep  # programmatic use
"""

import argparse
import csv
import itertools
from pathlib import Path

import run_config

# ── Registered model names ────────────────────────────────────────────────────
MODELS = list(run_config.MODELS.keys())

# ── Augmentation toggle keys (everything except the non-boolean "save_augmented") ──
_AUG_TOGGLE_KEYS = [
    k for k, v in run_config.AUGMENTATION.items()
    if isinstance(v, dict)  # each toggle is a dict; "save_augmented" is a bool
]

# ── Four fixed augmentation presets ──────────────────────────────────────────
_PHOTOMETRIC = {"horizontal_flip", "random_brightness_contrast",
                "hue_saturation_value", "random_gamma", "clahe"}
_GEOMETRIC   = {"horizontal_flip", "shift_scale_rotate", "coarse_dropout"}
_FULL        = set(_AUG_TOGGLE_KEYS)

AUG_PRESETS = {
    "none":        {k: False for k in _AUG_TOGGLE_KEYS},
    "photometric": {k: (k in _PHOTOMETRIC) for k in _AUG_TOGGLE_KEYS},
    "geometric":   {k: (k in _GEOMETRIC)   for k in _AUG_TOGGLE_KEYS},
    "full":        {k: True                for k in _AUG_TOGGLE_KEYS},
}

# ── Component keys ────────────────────────────────────────────────────────────
_COMP_KEYS = list(run_config.COMPONENTS.keys())


def generate_combinations():
    """Return a list of dicts, one per sweep run."""
    runs = []
    for model, (preset_name, aug_flags) in itertools.product(
        MODELS, AUG_PRESETS.items()
    ):
        # All 2^N on/off combinations for components
        for bits in itertools.product([False, True], repeat=len(_COMP_KEYS)):
            comp_state = dict(zip(_COMP_KEYS, bits))

            # Build experiment tag: sweep__<model>__aug_<preset>__k=0/1...
            comp_part = "__".join(
                f"{k}={'1' if v else '0'}" for k, v in comp_state.items()
            )
            tag = f"sweep__{model}__aug_{preset_name}__{comp_part}"

            runs.append({
                "model":          model,
                "aug_preset":     preset_name,
                "components":     comp_state,
                "experiment_tag": tag,
            })
    return runs


def run_combination(combo, idx=None, epochs=20):
    """Apply *combo* to run_config in-memory and execute one training run.

    Parameters
    ----------
    combo : dict
        One entry from generate_combinations().
    idx : int, optional
        Position in the sweep list (used only for error logging).
    """
    import run as _run  # local import avoids issues if run.py is not always present

    # 1. Disable every model, then enable only the one in this combo.
    for name in run_config.MODELS:
        run_config.MODELS[name]["enabled"] = (name == combo["model"])

    # 2. Write augmentation enabled-flags.
    #    AUG_PRESETS[preset_name] maps each toggle key → bool.
    aug_flags = AUG_PRESETS[combo["aug_preset"]]
    for key, enabled in aug_flags.items():
        run_config.AUGMENTATION[key]["enabled"] = enabled

    # 3. Write component toggles (each value is a plain bool in COMPONENTS).
    for key, enabled in combo["components"].items():
        run_config.COMPONENTS[key] = enabled

    # 4. Apply sweep-level training overrides.
    run_config.TRAINING["epochs"] = epochs
    run_config.TRAINING["experiment_tag"] = combo["experiment_tag"]

    # 5. Run the pipeline, logging any failure without aborting the sweep.
    #    On Windows the default console encoding (cp1252) cannot represent the
    #    Unicode box-drawing characters printed by run.py, so reconfigure stdout
    #    to UTF-8 for the duration of the call.
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    try:
        _run.main()
    except Exception:
        import traceback
        idx_str = str(idx) if idx is not None else "?"
        with open("sweep_errors.log", "a") as _err:
            _err.write(f"[combo {idx_str}] tag={combo['experiment_tag']}\n")
            traceback.print_exc(file=_err)
            _err.write("\n")


_RESULTS_CSV = "sweep_results.csv"
_CSV_FIELDNAMES = (
    ["run_index", "model", "aug_preset"]
    + _COMP_KEYS
    + ["experiment_tag", "final_val_metric", "epochs_completed"]
)


def _read_metrics(model_name, tag):
    """Return (final_val_loss_total, epochs_completed) from the Lightning CSV log.

    Uses the highest-numbered version_N directory so reruns don't return stale data.
    Returns (None, None) if the log file is missing or contains no validation rows.
    """
    log_base = Path(f"experiments/{model_name}/{tag}")
    version_dirs = sorted(
        log_base.glob("version_*"),
        key=lambda p: int(p.name.split("_")[1]),
    )
    if not version_dirs:
        return None, None

    csv_path = version_dirs[-1] / "metrics.csv"
    if not csv_path.exists():
        return None, None

    final_val = None
    max_epoch = None

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epoch_str = row.get("epoch", "").strip()
            if epoch_str:
                try:
                    epoch = int(float(epoch_str))
                    if max_epoch is None or epoch > max_epoch:
                        max_epoch = epoch
                except ValueError:
                    pass

            val_str = row.get("val/loss_total", "").strip()
            if val_str:
                try:
                    final_val = float(val_str)
                except ValueError:
                    pass

    epochs_completed = (max_epoch + 1) if max_epoch is not None else None
    return final_val, epochs_completed


def _append_result(row):
    """Append *row* (a dict keyed by _CSV_FIELDNAMES) to sweep_results.csv.

    Writes the header on the first call (when the file does not yet exist),
    then flushes immediately so results are visible even if the sweep is
    interrupted mid-run.
    """
    write_header = not Path(_RESULTS_CSV).exists()
    with open(_RESULTS_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
        f.flush()


def sweep(combos=None, epochs=20):
    """Run every combination in *combos* and log results to sweep_results.csv.

    Parameters
    ----------
    combos : list[dict] | None
        Combinations to run.  Defaults to the full list from generate_combinations().
    epochs : int
        Number of training epochs per combination (overrides the 20-epoch default).
    """
    if combos is None:
        combos = generate_combinations()

    for idx, combo in enumerate(combos):
        run_combination(combo, idx=idx, epochs=epochs)

        final_val, epochs_completed = _read_metrics(
            combo["model"], combo["experiment_tag"]
        )

        row = {
            "run_index":        idx,
            "model":            combo["model"],
            "aug_preset":       combo["aug_preset"],
            **{k: int(combo["components"][k]) for k in _COMP_KEYS},
            "experiment_tag":   combo["experiment_tag"],
            "final_val_metric": final_val,
            "epochs_completed": epochs_completed,
        }
        _append_result(row)
        print(
            f"[sweep {idx}] tag={combo['experiment_tag']} "
            f"val={final_val} epochs={epochs_completed}"
        )


def _print_table(runs):
    n_models = len(MODELS)
    n_presets = len(AUG_PRESETS)
    n_comp_combos = 2 ** len(_COMP_KEYS)
    expected = n_models * n_presets * n_comp_combos

    col_w = max(len(r["experiment_tag"]) for r in runs) + 2
    header = f"{'#':<6}{'model':<16}{'aug_preset':<14}{'components':<28}experiment_tag"
    print(header)
    print("-" * len(header))
    for i, r in enumerate(runs, 1):
        comp_str = str(r["components"])
        print(f"{i:<6}{r['model']:<16}{r['aug_preset']:<14}{comp_str:<28}{r['experiment_tag']}")

    print()
    print(f"Total rows : {len(runs)}")
    print(f"Expected   : {n_models} models × {n_presets} aug presets × 2^{len(_COMP_KEYS)} component combos = {expected}")
    assert len(runs) == expected, f"Row count mismatch: {len(runs)} != {expected}"
    print("OK — counts match.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sweep combination generator")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print all combinations and exit without training")
    parser.add_argument("--sweep", action="store_true",
                        help="Run every combination and log results to sweep_results.csv")
    parser.add_argument("--n", type=int, default=None,
                        help="Limit the sweep to the first N combinations")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Training epochs per combination (default: 20)")
    args = parser.parse_args()

    if args.dry_run:
        runs = generate_combinations()
        _print_table(runs)
    elif args.sweep:
        runs = generate_combinations()
        if args.n is not None:
            runs = runs[: args.n]
        sweep(runs, epochs=args.epochs)
