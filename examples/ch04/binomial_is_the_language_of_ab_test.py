"""A/Bテストの「CV数」は二項分布そのもの。10,000回引いて pmf と重ねる。

n=1,000 人に配って何人がコンバージョンしたかは、成功確率 p の独立な試行を n 回
足したもの、つまり $\\mathrm{Bin}(n, p)$ である。A/Bテストの分析で出てくる式は、
ほぼすべてこの1行から出てくる。

ここでは lift=0（AとBに差がない）の A/Bテストを 10,000 回まわし、A群のCV数の
ヒストグラムが二項分布の pmf と重なることを数え上げる。差がないのに毎回ちがう数字が
出る、その「ちがい方」の形が二項分布だと分かれば、第7章以降の検定は「この形から見て
珍しいか」を測る話になる。

    uv run python examples/ch04/binomial_is_the_language_of_ab_test.py
"""

import numpy as np
from scipy import stats

from toukei_tashikame import datasets, plots, sim

N_USER = 1_000
P_TRUE = 0.03
TRIALS = 10_000
SEED = 12


def one_trial(rng) -> tuple[int, int]:
    """A/Bテストを1回まわし、A群とB群のCV数を返す。lift=0 なので両者は同じ法則。"""
    d = datasets.ab_test(
        n_a=N_USER, n_b=N_USER, p_a=P_TRUE, lift=0.0,
        seed=int(rng.integers(2**31)),
    )
    return int(d.a.sum()), int(d.b.sum())


def main() -> None:
    counts = sim.repeat(one_trial, trials=TRIALS, seed=SEED, progress=False)
    a, b = counts[:, 0], counts[:, 1]

    print(f"--- {TRIALS:,} 回の A/Bテスト（各群 {N_USER:,} 人、真のCVR {P_TRUE}、lift=0）---")
    print(f"  A群のCV数  平均 {a.mean():.2f}   分散 {a.var(ddof=1):.2f}")
    print(f"  B群のCV数  平均 {b.mean():.2f}   分散 {b.var(ddof=1):.2f}")
    print(f"  理論       平均 {N_USER * P_TRUE:.1f}   分散 "
          f"{N_USER * P_TRUE * (1 - P_TRUE):.2f}   （np と np(1-p)）")
    print(f"  差がないのに A と B のCV数は {int(np.abs(a - b).mean())} 人ほど食い違う"
          f"（|A−B| の平均 {np.abs(a - b).mean():.2f} 人、最大 {int(np.abs(a - b).max())} 人）")

    dist = stats.binom(n=N_USER, p=P_TRUE)
    ks = np.arange(0, 70)
    emp = np.array([(a == k).mean() for k in ks])
    pmf = dist.pmf(ks)
    print("\n--- 経験分布 vs 二項分布の pmf ---")
    print(f"{'CV数 k':>8}{'実測の割合':>12}{'pmf':>10}{'差':>10}")
    for k in (20, 25, 30, 35, 40):
        print(f"{k:>8}{emp[k]:>12.4f}{pmf[k]:>10.4f}{emp[k] - pmf[k]:>+10.4f}")
    print(f"  最大差 {np.abs(emp - pmf).max():.4f}（k={int(np.abs(emp - pmf).argmax())} で）"
          f" / 全変動距離 {0.5 * np.abs(emp - pmf).sum():.4f}")

    lo, hi = dist.ppf(0.025), dist.ppf(0.975)
    inside = ((a >= lo) & (a <= hi)).mean()
    print(f"\n  二項分布の中央95% = [{lo:.0f}, {hi:.0f}] 人。実測がここに入った割合 {inside:.4f}")
    print(f"  差のないA/Bテストでも、観測CVRは {lo / N_USER:.3%}〜{hi / N_USER:.3%} の"
          "幅で動く。これが「効果なし」の見え方")

    plots.setup()
    fig, ax = plots.figure()
    ax.bar(ks, emp, width=0.9, color=plots.PALETTE["data"], alpha=0.55, lw=0,
           label="シミュレーション")
    ax.plot(ks, pmf, color=plots.PALETTE["truth"], lw=1.2, ls="--", dashes=(4, 2.0), zorder=5)
    ax.annotate("Bin(1000, 0.03) の pmf", xy=(38, dist.pmf(38)), fontsize=6.0,
                color=plots.PALETTE["truth"], ha="left", va="bottom")
    plots.mark_truth(ax, N_USER * P_TRUE, "np = 30")
    ax.set_xlabel("A群のCV数")
    ax.set_ylabel("割合")
    ax.set_title(f"lift=0 の A/Bテストを {TRIALS:,} 回")
    ax.set_xlim(10, 55)
    plots.save(fig, "fig-4-2-binomial-cv-counts.png")


if __name__ == "__main__":
    main()
