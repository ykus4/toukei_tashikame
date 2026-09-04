"""点推定値は「1回引いたくじ」にすぎない、を数え上げる。

μ=50, σ=10 の母集団から n=20 を引いて標本平均を作る、を 10,000 回。1回目と2回目で
値が違うこと、10,000本を並べると中心 μ・ばらつき σ/√n の分布になることを見る。
手元にある「推定値 51.83」は、この分布から1回引いた結果であって、μ そのものではない。

    uv run python examples/ch06/point_estimator_is_random.py
"""

import numpy as np
from scipy import stats

from toukei_tashikame import plots, sim

MU, SIGMA, N = 50.0, 10.0, 20
TRIALS, SEED = 10_000, 22


def one_trial(rng) -> float:
    """標本を1つ引いて、その標本平均（=点推定値）を返す。"""
    return float(rng.normal(MU, SIGMA, size=N).mean())


def main() -> None:
    plots.setup()
    means = sim.repeat(one_trial, trials=TRIALS, seed=SEED, progress=False)
    theory_se = SIGMA / np.sqrt(N)

    print("--- 1本ずつ見る（同じ手続き・同じ母集団・違う標本）---")
    for i in range(3):
        print(f"  {i + 1} 回目の点推定値  {means[i]:.2f}   （真値 {MU:g} との差 {means[i] - MU:+.2f}）")

    print(f"\n--- {TRIALS:,} 本を並べる ---")
    print(f"  推定値の平均   {means.mean():.4f}   （真値 {MU:g}。ずれ {means.mean() - MU:+.4f}）")
    print(f"  推定値のSD     {means.std(ddof=1):.4f}   （理論 σ/√n = {theory_se:.4f}）")

    err = means - MU
    lo, hi = np.quantile(err, [0.05, 0.95])
    print(f"  外し幅の90%範囲 [{lo:+.2f}, {hi:+.2f}]   （幅の半分 {(hi - lo) / 2:.2f}）")
    print(f"  1本だけ見て {np.abs(err).mean():.2f} 程度ずれているのが普通、と読む")

    fig, ax = plots.figure()
    grid = np.linspace(means.min(), means.max(), 400)
    plots.sim_hist(ax, means, theory=(grid, stats.norm.pdf(grid, MU, theory_se)),
                   theory_label=f"N({MU:g}, {theory_se:.2f}²)")
    plots.mark_truth(ax, MU, f"真の μ = {MU:g}")
    ax.set_xlabel(f"標本平均（n={N}）")
    ax.set_ylabel("密度")
    ax.set_title(f"点推定値そのものが分布する（{TRIALS:,} 回）")
    plots.save(fig, "fig-6-1-estimator-distribution.png")


if __name__ == "__main__":
    main()
