"""帰無仮説が正しいとき、p値は一様分布になる。

「p = 0.03 は 0.7 より珍しい」と読みたくなるが、帰無仮説が真の世界では、p値は
0 から 1 のどこにも同じ確率で落ちる。0.03 が出るのも 0.73 が出るのも同じくらい
ありふれている。α=5% で棄却する、という手続きが誤警報を 5% に抑えるのは、
まさにこの一様性のおかげである。

差がまったく無いデータで検定を10,000回まわしてp値を集め、ヒストグラムが平らに
なること、KS検定が一様分布からのずれを見つけられないことを確かめる。

    uv run python examples/ch07/pvalue_is_uniform.py
"""

import numpy as np
from scipy import stats

from toukei_tashikame import plots, sim, testing

N = 30
TRIALS = 10_000
ALPHA = 0.05


def null_pvalue(rng) -> float:
    """帰無仮説が正しい（μ=0）標本を引いて、1標本t検定のp値を返す。"""
    x = rng.normal(0.0, 1.0, size=N)
    return testing.t_1samp(x, 0.0).pvalue


def main() -> None:
    plots.setup()

    p = sim.repeat(null_pvalue, trials=TRIALS, seed=7, progress=False)

    ks = stats.kstest(p, "uniform")
    rate = float((p < ALPHA).mean())
    se = np.sqrt(rate * (1 - rate) / TRIALS)

    print(f"--- 差ゼロのデータで {TRIALS:,} 回検定（n={N}, seed=7）---")
    print(f"  p値の平均      {p.mean():.4f}   （一様分布なら 0.5）")
    print(f"  p値の中央値    {np.median(p):.4f}   （一様分布なら 0.5）")
    print(f"  p値のSD        {p.std(ddof=1):.4f}   （一様分布なら {1 / np.sqrt(12):.4f}）")
    print("\n  十分位ごとの個数（一様なら各 1,000）")
    counts, _ = np.histogram(p, bins=10, range=(0.0, 1.0))
    print("   " + "  ".join(f"{c:,}" for c in counts))
    print(f"\n  一様分布とのKS検定    D = {ks.statistic:.4f}, p = {ks.pvalue:.4f}")
    print("  ← このp値が大きい = 「一様でない」とは言えない")
    print(f"\n  p < {ALPHA} だった割合  {rate:.4f} ± {1.96 * se:.4f}"
          f"   （設計値 {ALPHA}）")
    print("  一様だからこそ、下から5%を切り取る手続きが誤警報を5%に抑える")

    # --- 図: ヒストグラムと累積分布 ---
    fig, axes = plots.figure(1, 2)
    ax = axes[0]
    ax.hist(p, bins=20, range=(0, 1), density=True, color=plots.PALETTE["data"],
            alpha=0.55, lw=0)
    ax.axhline(1.0, color=plots.PALETTE["truth"], lw=1.1, zorder=5)
    ax.axvspan(0, ALPHA, color=plots.PALETTE["reject"], alpha=0.55, lw=0, zorder=1)
    ax.set_ylim(0, 1.45)
    ax.annotate("一様分布の密度 = 1", xy=(0.03, 1.06), fontsize=6.0,
                color=plots.PALETTE["truth"], va="bottom")
    ax.annotate(f"p < {ALPHA}\n= 面積の {rate:.1%}", xy=(ALPHA, 0.55),
                xytext=(8, 0), textcoords="offset points", fontsize=6.0,
                color=plots.PALETTE["reject"])
    ax.set_xlabel("p値")
    ax.set_ylabel("密度")
    ax.set_title(f"帰無が真のときのp値（{TRIALS:,}回）")

    ax = axes[1]
    xs = np.sort(p)
    ax.plot(xs, np.arange(1, TRIALS + 1) / TRIALS, color=plots.PALETTE["estimate"], lw=1.2)
    ax.plot([0, 1], [0, 1], color=plots.PALETTE["truth"], lw=1.0, ls="--", dashes=(4, 2.0))
    ax.annotate("y = x（一様）", xy=(0.55, 0.42), fontsize=6.0, color=plots.PALETTE["truth"])
    ax.set_xlabel("p値")
    ax.set_ylabel("累積割合")
    ax.set_title(f"KS検定 p = {ks.pvalue:.4f}")
    fig.tight_layout()
    plots.save(fig, "fig-7-4-pvalue-uniform.png")


if __name__ == "__main__":
    main()
