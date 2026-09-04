"""p値を、公式ではなく数え上げで作る。

p値は「帰無仮説が正しいとして、観測した以上に極端なことが起きる確率」である。この
定義文はそのまま手続きになる。帰無仮説（μ=0）が本当に成り立つ世界を10,000回作り、
そこで出た t 統計量のうち、手元の t より極端だったものを数えればよい。

割り算の結果は ``scipy.stats.ttest_1samp`` が返す p 値と一致する。t 分布の式は、
この数え上げを紙の上で先にやっておいたものにすぎない。

    uv run python examples/ch07/pvalue_by_counting.py
"""

import numpy as np
from scipy import stats

from toukei_tashikame import plots, sim, testing

N = 20
MU_TRUE = 0.7      # 観測データを作った真の効果。検定はこれを知らない
MU_NULL = 0.0      # 帰無仮説が主張する値
TRIALS = 10_000


def null_t(rng) -> float:
    """帰無仮説が正しい世界で、n=20 の標本から t 統計量を1つ作る。

    母標準偏差は 1 にしてあるが、t は尺度に依らない（σ を何にしても分布は同じ）ので、
    ここで選んだ 1 は結果に効かない。
    """
    x = rng.normal(MU_NULL, 1.0, size=N)
    return (x.mean() - MU_NULL) / (x.std(ddof=1) / np.sqrt(N))


def main() -> None:
    plots.setup()

    # --- 手元にある1つの標本 ---
    x = np.random.default_rng(7).normal(MU_TRUE, 1.0, size=N)
    obs = testing.t_1samp(x, MU_NULL)
    t_obs = obs.stat
    print("--- 観測した標本（n=20, seed=7）---")
    print(f"  標本平均 {x.mean():.4f}  標本SD {x.std(ddof=1):.4f}")
    print(f"  t = {t_obs:.4f}")

    # --- 帰無仮説が正しい世界を10,000回作る ---
    t_null = sim.repeat(null_t, trials=TRIALS, seed=7, progress=False)
    extreme = np.abs(t_null) >= abs(t_obs)
    p_count = float(extreme.mean())
    se = np.sqrt(p_count * (1 - p_count) / TRIALS)

    print(f"\n--- μ=0 の世界を {TRIALS:,} 回作って数える ---")
    print(f"  |t| >= {abs(t_obs):.4f} だった回数   {extreme.sum():,} / {TRIALS:,}")
    print(f"  数え上げのp値                {p_count:.4f} ± {1.96 * se:.4f}")
    print(f"  ttest_1samp のp値            {stats.ttest_1samp(x, MU_NULL).pvalue:.4f}")
    print(f"  testing.t_1samp のp値        {obs.pvalue:.4f}")
    print("\n  同じものを2通りで出しただけ。t 分布の式は、この数え上げを"
          "紙の上で先に済ませたもの")

    # --- 図: 帰無分布と、観測した t より外側 ---
    fig, ax = plots.figure()
    grid = np.linspace(-5, 5, 400)
    plots.null_vs_alt(ax, grid, stats.t.pdf(grid, df=N - 1), crit=abs(t_obs), tail="upper")
    ax.fill_between(grid[grid <= -abs(t_obs)], stats.t.pdf(grid[grid <= -abs(t_obs)], df=N - 1),
                    color=plots.PALETTE["reject"], alpha=0.55, lw=0, zorder=2)
    ax.hist(t_null, bins=60, density=True, color=plots.PALETTE["data"], alpha=0.35, lw=0)
    ax.annotate(f"観測 t = {t_obs:.2f}", xy=(abs(t_obs), 0.30), xycoords=("data", "axes fraction"),
                xytext=(6, 0), textcoords="offset points", fontsize=6.2,
                color=plots.PALETTE["reject"])
    ax.annotate(f"この2つの面積の合計が p = {p_count:.4f}", xy=(0.02, 0.86),
                xycoords="axes fraction", fontsize=6.2, color=plots.PALETTE["ink2"])
    ax.set_xlabel("帰無仮説が正しい世界での t 統計量")
    ax.set_ylabel("密度")
    ax.set_title("p値 = 帰無分布のうち、観測より外側の面積")
    plots.save(fig, "fig-7-3-pvalue-by-counting.png")


if __name__ == "__main__":
    main()
