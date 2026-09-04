"""歪度と尖度は、正規分布から引いてもよく暴れる。

真の歪度 0・真の超過尖度 0 の N(0,1) から n=50 を 10,000 回引き、標本歪度・標本尖度が
どこまで散らばるかを数える。3乗・4乗を平均する統計量なので、端の1〜2点で値が決まって
しまい、n=50 程度では ±0.7 くらいは平気で動く。

「歪度が 0.5 だから右に歪んでいる」と言う前に、この図の幅を見ておく。正規分布から
引いただけでその程度の値は出る。手書き実装（``describe``）と ``scipy.stats`` の
一致もここで確認しておく。

    uv run python examples/ch01/skewness_kurtosis_sampling_noise.py
"""

import numpy as np
from scipy import stats

from toukei_tashikame import describe, plots, sim

N = 50
TRIALS = 10_000


def one_trial(rng):
    """N(0,1) から n=50 を引いて、標本歪度と標本超過尖度を返す。"""
    x = rng.normal(0.0, 1.0, size=N)
    return describe.skewness(x), describe.kurtosis(x)


def main() -> None:
    plots.setup()

    v = sim.repeat(one_trial, trials=TRIALS, seed=0, progress=False)
    skew, kurt = v[:, 0], v[:, 1]

    s_lo, s_hi = np.percentile(skew, [2.5, 97.5])
    k_lo, k_hi = np.percentile(kurt, [2.5, 97.5])

    print(f"--- N(0,1) から n={N} を {TRIALS:,} 回（真の歪度 0 / 真の超過尖度 0）---")
    print(f"{'':<14}{'平均':>10}{'標準偏差':>10}{'95%範囲':>22}")
    print(f"{'標本歪度':<14}{skew.mean():>10.4f}{skew.std(ddof=1):>10.4f}"
          f"{f'[{s_lo:.3f}, {s_hi:.3f}]':>22}")
    print(f"{'標本超過尖度':<14}{kurt.mean():>10.4f}{kurt.std(ddof=1):>10.4f}"
          f"{f'[{k_lo:.3f}, {k_hi:.3f}]':>22}")
    print(f"\n  理論上の標準偏差（大標本近似）  歪度 √(6/n)={np.sqrt(6 / N):.4f} / "
          f"超過尖度 √(24/n)={np.sqrt(24 / N):.4f}")
    print(f"  |歪度| > 0.5 が出る割合  {float((np.abs(skew) > 0.5).mean()):.4f}"
          "  ← 正規分布から引いただけでこれだけ出る")
    print(f"  超過尖度は右に歪む（歪度の歪度）。上側の裾のほうが長い: "
          f"下側 {k_lo:.3f} / 上側 {k_hi:.3f}")

    # 手書き実装（describe）と scipy の照合。同じ標本を両方に通す。
    rng = np.random.default_rng(99)
    ds, dk = [], []
    for _ in range(200):
        x = rng.normal(size=N)
        ds.append(abs(describe.skewness(x) - stats.skew(x, bias=False)))
        dk.append(abs(describe.kurtosis(x) - stats.kurtosis(x, bias=False)))
    print(f"\n  describe vs scipy（200標本の最大絶対差）  歪度 {max(ds):.2e} / "
          f"尖度 {max(dk):.2e}")

    fig, axes = plots.figure(1, 2, w=1.6)
    for ax, val, lo, hi, name in (
        (axes[0], skew, s_lo, s_hi, "標本歪度"),
        (axes[1], kurt, k_lo, k_hi, "標本超過尖度"),
    ):
        ax.hist(val, bins=60, color=plots.PALETTE["data"], alpha=0.55, lw=0)
        plots.mark_truth(ax, 0.0, "真値 = 0")
        for b in (lo, hi):
            ax.axvline(b, color=plots.PALETTE["estimate"], lw=0.9, ls="--", dashes=(4, 2.0))
        ax.annotate(f"95%範囲 [{lo:.2f}, {hi:.2f}]", xy=(0.5, 0.06), xycoords="axes fraction",
                    fontsize=6.0, color=plots.PALETTE["estimate"], ha="center")
        ax.set_xlabel(name)
        ax.set_ylabel("回数")
        ax.set_title(f"{name}（n={N}）")

    plots.save(fig, "fig-1-5-shape-sampling.png")


if __name__ == "__main__":
    main()
