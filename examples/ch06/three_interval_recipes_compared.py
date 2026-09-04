"""同じ標本に3つのレシピ — 正規近似・t分布・ブートストラップ。

1つの n=20 の標本に3種類の区間を当てて並べ、そのあと各10,000回で被覆を数える。
どれも「95%」と名乗るが、仮定しているものが違う。正規近似は σ を知っているつもりで
使い、t は母集団の正規性を使い、ブートストラップは標本が母集団の縮図だと仮定する。
n=20 では、その差が被覆の数字に出る。

    uv run python examples/ch06/three_interval_recipes_compared.py
"""

import numpy as np

from toukei_tashikame import datasets, estimate, plots, sim

MU, SIGMA, N = 50.0, 10.0, 20
SAMPLE_SEED, TRIALS, SEED = 23, 10_000, 28
B = 1_000   # 被覆を数えるループの中の再標本数。表示用の1本だけ 10,000 にする


def boot_percentile(x, rng, b: int = B) -> tuple[float, float]:
    """percentile ブートストラップ区間。試行の rng から引くので毎回違う再標本になる。"""
    idx = rng.integers(0, x.size, size=(b, x.size))
    boots = x[idx].mean(axis=1)
    return tuple(np.quantile(boots, [0.025, 0.975]))


def main() -> None:
    plots.setup()
    x = datasets.normal_sample(N, mu=MU, sigma=SIGMA, seed=SAMPLE_SEED)
    recipes = {
        "正規近似 (z)": estimate.ci_mean_z(x),
        "t 分布": estimate.ci_mean_t(x),
        "ブートストラップ": estimate.boot_ci(x, B=10_000, kind="percentile", seed=SAMPLE_SEED),
    }

    print(f"--- 同じ1つの標本（n={N}, seed={SAMPLE_SEED}）に3つのレシピ ---")
    print(f"  標本平均 {x.mean():.4f} / 標本SD {x.std(ddof=1):.4f}（真値 μ={MU:g}, σ={SIGMA:g}）")
    for name, (lo, hi) in recipes.items():
        print(f"  {name:<16} [{lo:.2f}, {hi:.2f}]   幅 {hi - lo:.2f}")
    print("  幅の差は σ の推定の不確かさをどう埋め合わせるかの差。t が一番広い")

    print(f"\n--- 各 {TRIALS:,} 回で被覆を数える（名目 95%）---")
    trials_fn = {
        "正規近似 (z)": lambda rng: estimate.ci_mean_z(rng.normal(MU, SIGMA, size=N)),
        "t 分布": lambda rng: estimate.ci_mean_t(rng.normal(MU, SIGMA, size=N)),
        "ブートストラップ": lambda rng: boot_percentile(rng.normal(MU, SIGMA, size=N), rng),
    }
    results = {}
    for name, fn in trials_fn.items():
        res = sim.coverage(fn, truth=MU, trials=TRIALS, seed=SEED, progress=False)
        results[name] = res
        width = float(np.mean(res.intervals[:, 1] - res.intervals[:, 0]))
        print(f"  {name:<16} 被覆 {res.rate:.4f} ± {1.96 * res.se:.4f}   平均幅 {width:.2f}")
    print(f"  ブートストラップは B={B:,} の再標本で作った。名目に届かないのは"
          "「標本が母集団の縮図」が n=20 では苦しいから")
    print("  t だけが 0.95 に乗る。正規性が本当に成り立っている場面では t が正解である")

    fig, ax = plots.figure(h=0.9)
    for i, (name, (lo, hi)) in enumerate(reversed(recipes.items())):
        y = i
        plots.mark_interval(ax, lo, hi, y=y)
        ax.plot([x.mean()], [y], "o", ms=3.5, color=plots.PALETTE["estimate"], zorder=5)
        cov = results[name].rate
        ax.annotate(f"{name}  幅 {hi - lo:.2f} / 被覆 {cov:.4f}", xy=((lo + hi) / 2, y + 0.12),
                    ha="center", va="bottom", fontsize=6.0, color=plots.PALETTE["ink2"])
    plots.mark_truth(ax, MU, f"真値 μ = {MU:g}")
    ax.set_ylim(-0.6, len(recipes) - 0.3)
    ax.set_yticks([])
    ax.set_xlabel("μ の 95% 区間")
    ax.set_title(f"同じ標本・3つのレシピ（n={N}）")
    plots.save(fig, "fig-6-6-three-intervals.png")


if __name__ == "__main__":
    main()
