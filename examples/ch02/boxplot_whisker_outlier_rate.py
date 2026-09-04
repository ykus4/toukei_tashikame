"""箱ひげ図の「外れ値」は、外れていなくても出る。1.5×IQR の実際の率を数える。

正規分布から引いただけの、何も汚れていないデータでも、1.5×IQR の外には一定の割合で
点が落ちる。理論値は約 0.70%。n=100 なら1試行あたり0.7個で、「1個以上出る」試行は
半分近くになる。箱ひげ図の点を見て「外れ値がある」と言う前に、この率を知っておく。

    uv run python examples/ch02/boxplot_whisker_outlier_rate.py
"""

import numpy as np
from scipy import stats

from toukei_tashikame import plots, sim

N, TRIALS, SEED = 100, 10_000, 0


def fences(x: np.ndarray) -> tuple[float, float]:
    """Tukey のひげの先。Q1 − 1.5 IQR と Q3 + 1.5 IQR。"""
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr


def one_trial(rng) -> tuple[int, int]:
    """標本を1つ引いて、(ひげの外に出た点の数, 1個以上出たか) を返す。"""
    x = rng.normal(0.0, 1.0, size=N)
    lo, hi = fences(x)
    k = int(((x < lo) | (x > hi)).sum())
    return k, int(k > 0)


def theoretical_rate() -> tuple[float, float]:
    """母集団が正規なら、ひげの先は母分位点で決まる。その外側の確率。"""
    q1, q3 = stats.norm.ppf(0.25), stats.norm.ppf(0.75)
    fence = q3 + 1.5 * (q3 - q1)
    return fence, float(2 * stats.norm.sf(fence))


def main() -> None:
    plots.setup()
    print("--- 2-4 ひげの外に出る点の割合 ---")

    fence, p_theory = theoretical_rate()
    print(f"母集団 N(0,1) の Q1={stats.norm.ppf(0.25):.4f} / Q3={stats.norm.ppf(0.75):.4f} "
          f"/ IQR={stats.norm.ppf(0.75) - stats.norm.ppf(0.25):.4f}")
    print(f"ひげの先は ±{fence:.4f}。その外側の確率は {p_theory:.5f}"
          f"（{100 * p_theory:.3f}%）  ← 理論値")

    with sim.Timer(f"n={N} を {TRIALS:,} 回"):
        out = sim.repeat(one_trial, trials=TRIALS, seed=SEED, progress=False)
    counts = out[:, 0].astype(int)
    any_out = out[:, 1].astype(bool)

    point_rate = counts.sum() / (TRIALS * N)
    point_se = np.sqrt(point_rate * (1 - point_rate) / (TRIALS * N))
    trial_rate = float(any_out.mean())
    trial_se = np.sqrt(trial_rate * (1 - trial_rate) / TRIALS)

    print(f"\n点ベースの外れ値率      {point_rate:.4f} ± {1.96 * point_se:.4f}"
          f"   （理論 {p_theory:.5f}）")
    print(f"1個以上出た試行の割合   {trial_rate:.4f} ± {1.96 * trial_se:.4f}")
    print(f"1試行あたりの平均個数   {counts.mean():.4f} 個   （理論 {N * p_theory:.4f} 個）")
    print(f"最大は {counts.max()} 個。中央値は {int(np.median(counts))} 個")

    print("\n  1試行に出た個数の分布:")
    for k in range(0, 6):
        share = float((counts == k).mean())
        bar = "█" * round(share * 50)
        print(f"    {k}個 {share:6.4f} {bar}")
    print(f"    6個以上 {float((counts >= 6).mean()):6.4f}")

    # 二項近似（各点が独立に確率 p で外れると見なす）と突き合わせる。
    print(f"\n  二項近似 Binom(n={N}, p={p_theory:.5f}) で「0個」の確率 "
          f"{(1 - p_theory) ** N:.4f} → 1個以上は {1 - (1 - p_theory) ** N:.4f}")
    print("  ← 実測の方が高い。ひげの位置を母集団から知っているのではなく、"
          "\n    同じ標本の四分位数から推定しているため。IQR が小さめに出た試行では"
          "\n    基準が締まって外れ値が一気に増え（最大 "
          f"{counts.max()} 個）、分布は二項より裾が重くなる")

    fig, axes = plots.figure(1, 2, h=1.0, w=2.0)
    ax = axes[0]
    demo = np.random.default_rng(SEED).normal(0.0, 1.0, size=N)
    lo, hi = fences(demo)
    ax.boxplot([demo], orientation="horizontal", widths=0.5, whis=1.5,
               flierprops={"marker": "o", "markersize": 2.5,
                           "markerfacecolor": plots.PALETTE["reject"],
                           "markeredgecolor": "none"},
               medianprops={"color": plots.PALETTE["truth"]},
               boxprops={"color": plots.PALETTE["data"]},
               whiskerprops={"color": plots.PALETTE["data"]},
               capprops={"color": plots.PALETTE["data"]})
    ax.axvline(lo, color=plots.PALETTE["reject"], lw=0.8, ls="--", dashes=(3, 2))
    ax.axvline(hi, color=plots.PALETTE["reject"], lw=0.8, ls="--", dashes=(3, 2))
    ax.set_title(f"1試行の例（n={N}、ひげの外は {int(((demo < lo) | (demo > hi)).sum())} 個）")
    ax.set_yticks([])
    ax.set_xlabel("x")

    ax = axes[1]
    ks = np.arange(0, counts.max() + 1)
    ax.bar(ks, [float((counts == k).mean()) for k in ks], color=plots.PALETTE["data"],
           alpha=0.65, lw=0)
    ax.plot(ks, stats.binom.pmf(ks, N, p_theory), color=plots.PALETTE["truth"],
            lw=1.2, ls="--", dashes=(4, 2), zorder=5)
    ax.annotate(f"二項近似 Binom({N}, {p_theory:.4f})", xy=(0.55, 0.8),
                xycoords="axes fraction", fontsize=6.0, color=plots.PALETTE["truth"])
    ax.set_title(f"{TRIALS:,}試行での外れ値の個数")
    ax.set_xlabel("1試行に出た外れ値の個数")
    ax.set_ylabel("割合")
    plots.save(fig, "fig-2-4-whisker-rule.png")


if __name__ == "__main__":
    main()
