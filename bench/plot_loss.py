"""Plot train/val loss curve from a run's loss_log.csv.

Usage: python bench/plot_loss.py bench/run1
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def main():
    run_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "bench/run1")
    df = pd.read_csv(run_dir / "loss_log.csv")

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(9, 5))

    sns.lineplot(data=df, x="step", y="train_loss", label="train_loss", ax=ax)
    sns.lineplot(data=df, x="step", y="val_loss", label="val_loss", ax=ax)

    ax.axhline(np.log(65), color="gray", linestyle="--", linewidth=1, label="ln(65) — untrained baseline")

    ax.set_xlabel("step")
    ax.set_ylabel("cross-entropy loss")
    ax.set_title(f"{run_dir.name} — train / val loss")
    ax.legend()

    fig.tight_layout()
    out_path = run_dir / "loss_curve.png"
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
