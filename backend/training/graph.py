"""Plot the REAL training curves from the TensorBoard event files.

The previous version crashed (it used `training_accuracy` on the line above
its own definition) and its numbers were hardcoded placeholders labelled
"# Example values". Never quote those on a CV - run test.py instead.
"""
import glob
import os

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_scalars(logdir="logs"):
    series = {}
    for path in sorted(glob.glob(os.path.join(logdir, "events.out.tfevents.*"))):
        acc = EventAccumulator(path)
        acc.Reload()
        for tag in acc.Tags().get("scalars", []):
            series.setdefault(tag, []).extend(
                (e.step, e.value) for e in acc.Scalars(tag))
    for tag in series:
        series[tag].sort()
    return series


def main():
    s = load_scalars()
    if not s:
        raise SystemExit("No event files found in logs/ - run training.py first.")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    if "train/loss" in s:
        xs, ys = zip(*s["train/loss"])
        axes[0].plot(xs, ys, linewidth=0.8, label="training loss")
    if "eval/loss" in s:
        xs, ys = zip(*s["eval/loss"])
        axes[0].plot(xs, ys, "o-", label="validation loss")
    axes[0].axhline(0.693, ls="--", c="grey", lw=1,
                    label="ln(2) = random guess")
    axes[0].set_xlabel("step"); axes[0].set_ylabel("loss")
    axes[0].set_title("Loss"); axes[0].legend()

    plotted = False
    for tag in ("eval/f1", "eval/accuracy", "eval/precision", "eval/recall"):
        if tag in s:
            xs, ys = zip(*s[tag])
            axes[1].plot(xs, ys, "o-", label=tag.split("/")[1])
            plotted = True
    if not plotted:
        axes[1].text(0.5, 0.5,
                     "No accuracy/F1 logged.\nAdd compute_metrics and retrain.",
                     ha="center", va="center", transform=axes[1].transAxes)
    axes[1].set_xlabel("step"); axes[1].set_title("Metrics")
    if plotted:
        axes[1].legend()

    plt.tight_layout()
    plt.savefig("model_performance_over_epochs.png", dpi=150)
    print("wrote model_performance_over_epochs.png")


if __name__ == "__main__":
    main()
