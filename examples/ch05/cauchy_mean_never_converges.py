"""破れ目の極端な例 — コーシー分布では標本平均が永久に収束しない。

大数の法則も中心極限定理も「母平均と母分散が存在する」ことを前提にしている。コーシー
分布はどちらも持たない。だから n をいくら増やしても標本平均は落ち着かない。実際、
コーシーの標本平均の分布は**n によらずコーシーそのもの**で、1個の観測と100万個の平均が
同じばらつきを持つ。

一方で標本中央値はちゃんと収束する。「平均が使えない」ことと「何も言えない」ことは
違う——裾の重いデータで中央値や分位点を使うのは、この差に基づく実務判断である。

    uv run python examples/ch05/cauchy_mean_never_converges.py
"""

import unicodedata

import numpy as np

from toukei_tashikame import datasets, plots

N_MAX = 1_000_000
N_PATHS = 50
SEED = 20
CHECKPOINTS = [10, 100, 1_000, 10_000, 100_000, 1_000_000]


def rj(text: str, width: int) -> str:
    """全角を2桁として数えて右詰めする。日本語の見出しでも表の桁が揃う。"""
    w = sum(0 if unicodedata.combining(c) else
            2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in text)
    return " " * max(width - w, 0) + text


def main() -> None:
    plots.setup()
    master = np.random.default_rng(SEED)
    seeds = master.integers(0, 2**31 - 1, size=N_PATHS)

    keep_idx = np.unique(np.logspace(0, np.log10(N_MAX), 300).astype(int)) - 1
    mean_at, median_at, mean_paths = [], [], []
    for s in seeds:
        x = datasets.heavy_tailed(N_MAX, kind="cauchy", seed=int(s))
        running = np.cumsum(x) / np.arange(1, N_MAX + 1)
        mean_at.append(running[np.array(CHECKPOINTS) - 1])
        median_at.append([np.median(x[:c]) for c in CHECKPOINTS])
        mean_paths.append(running[keep_idx])
    mean_at = np.array(mean_at)          # (50, 6)
    median_at = np.array(median_at)
    mean_paths = np.array(mean_paths)

    print(f"--- 標準コーシー、{N_PATHS}本の軌跡（真の中心は 0）---")
    print("  " + rj("n", 9) + "  " + rj("|x̄| の中央値", 16)
          + "  " + rj("|x̄| の最大", 15) + "  " + rj("x̄ のSD", 12)
          + "  " + rj("|中央値| の中央値", 18))
    for j, n in enumerate(CHECKPOINTS):
        m = mean_at[:, j]
        med = median_at[:, j]
        print(f"  {n:>9,}  {np.median(np.abs(m)):>16.4f}  {np.abs(m).max():>15.2f}"
              f"  {m.std(ddof=1):>12.2f}  {np.median(np.abs(med)):>18.4f}")

    print("\n  ← 平均の列は n を10万倍にしても縮まない。|標本平均| の中央値が 1.0 前後に"
          "留まるのは、\n     標本平均の分布が n によらず標準コーシーのままだから"
          "（|コーシー| の中央値 = 1）")
    print("  ← 中央値の列だけが π/(2√n) の速さで 0 に縮んでいく")

    final_mean, final_med = mean_at[:, -1], median_at[:, -1]
    print(f"\n  n=10⁶ での標本平均: 絶対値の最大 {np.abs(final_mean).max():.2f}"
          f" / SD {final_mean.std(ddof=1):.2f}")
    print(f"  n=10⁶ での標本中央値: 平均 {final_med.mean():+.4f}"
          f" / SD {final_med.std(ddof=1):.4f}（理論 π/(2√n) = "
          f"{np.pi / (2 * np.sqrt(N_MAX)):.4f}）")
    worst = int(np.argmax(np.abs(final_mean)))
    jump = np.abs(mean_paths[worst]).max()
    print(f"  いちばん暴れた軌跡は {N_MAX:,} 個目までに |x̄| が {jump:.1f} まで飛んだ。"
          "1個の外れ値が全部を持っていく")

    fig, (ax1, ax2) = plots.figure(1, 2, w=2.0)
    xs = keep_idx + 1
    for p in mean_paths:
        ax1.plot(xs, p, color=plots.PALETTE["estimate"], lw=0.5, alpha=0.4)
    plots.mark_truth(ax1, 0.0, "分布の中心 = 0", axis="y")
    ax1.set_xscale("log")
    ax1.set_ylim(-6, 6)
    ax1.set_xlabel("n（対数）")
    ax1.set_ylabel("累積平均")
    ax1.set_title(f"標本平均は収束しない（{N_PATHS}本）")

    for m in median_at:
        ax2.plot(CHECKPOINTS, m, color=plots.PALETTE["estimate"], lw=0.5, alpha=0.4)
    plots.mark_truth(ax2, 0.0, "分布の中心 = 0", axis="y")
    ax2.set_xscale("log")
    ax2.set_ylim(-6, 6)
    ax2.set_xlabel("n（対数）")
    ax2.set_ylabel("累積中央値")
    ax2.set_title("標本中央値は収束する（同じ縦軸）")
    plots.save(fig, "fig-5-6-cauchy-never-converges.png")


if __name__ == "__main__":
    main()
