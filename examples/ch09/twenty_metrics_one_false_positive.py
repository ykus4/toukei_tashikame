"""指標を20本並べれば、効果がゼロでも6割の確率でどれかが有意になる。

ダッシュボードに指標が20個載っているA/Bテストで「有意になった指標がある」ことは、
ほとんど何も意味しない。1本あたり5%を20本ぶん見れば、少なくとも1本が有意になる確率は
$1-0.95^{20}=0.6415$ だからである。これは検定の欠陥ではなく、検定を20回やった当然の
帰結でしかない。

真の差がゼロの指標を20本作り、少なくとも1本が有意になる割合を 10,000 回で数え上げて、
$1-0.95^{m}$ の曲線と重ねる。

    uv run python examples/ch09/twenty_metrics_one_false_positive.py
"""

import numpy as np
from scipy import special, stats

from toukei_tashikame import plots, sim

M = 20              # 指標の本数
N = 100             # 群あたりの人数
ALPHA = 0.05
TRIALS = 10_000
SEED = 93


def one_trial(rng) -> np.ndarray:
    """20本ぶんの Welch 検定の p 値をまとめて返す。真の差はどれもゼロ。"""
    a = rng.normal(0.0, 1.0, size=(M, N))
    b = rng.normal(0.0, 1.0, size=(M, N))
    s1, s2 = a.var(axis=1, ddof=1) / N, b.var(axis=1, ddof=1) / N
    t = (a.mean(axis=1) - b.mean(axis=1)) / np.sqrt(s1 + s2)
    df = (s1 + s2) ** 2 / (s1**2 / (N - 1) + s2**2 / (N - 1))
    return 2 * special.stdtr(df, -np.abs(t))


def draw(observed: np.ndarray, counts: np.ndarray) -> None:
    ms = np.arange(1, M + 1)
    theory = 1 - (1 - ALPHA) ** ms

    fig, axes = plots.figure(1, 2, w=2.0, h=0.95)

    ax = axes[0]
    ax.plot(ms, theory, color=plots.PALETTE["truth"], lw=1.2, ls="--", dashes=(4, 2.0),
            zorder=5)
    ax.scatter(ms, observed, s=9, color=plots.PALETTE["estimate"], lw=0, zorder=6)
    ax.annotate("理論 $1-0.95^{m}$", xy=(ms[13], theory[13]), xytext=(2, -9),
                textcoords="offset points", fontsize=6.0, color=plots.PALETTE["truth"])
    ax.annotate("数え上げ", xy=(ms[5], observed[5]), xytext=(2, 8),
                textcoords="offset points", fontsize=6.0, color=plots.PALETTE["estimate"])
    ax.set_xlabel("並べた指標の本数 m")
    ax.set_ylabel("少なくとも1本が有意になる割合")
    ax.set_ylim(0, 0.75)
    ax.set_title("族全体の誤り（FWER）は本数で決まる")

    ax = axes[1]
    k = np.arange(0, M + 1)
    obs = np.array([(counts == i).mean() for i in k])
    ax.bar(k, obs, width=0.75, color=plots.PALETTE["data"], alpha=0.65, lw=0)
    ax.plot(k, stats.binom.pmf(k, M, ALPHA), color=plots.PALETTE["truth"], lw=1.2,
            ls="--", dashes=(4, 2.0), marker="o", ms=2.2, zorder=5)
    ax.annotate("二項分布 $B(20, 0.05)$", xy=(3.0, stats.binom.pmf(3, M, ALPHA)),
                xytext=(6, 6), textcoords="offset points", fontsize=6.0,
                color=plots.PALETTE["truth"])
    ax.set_xlim(-0.6, 6.6)
    ax.set_xlabel("有意になった指標の本数")
    ax.set_ylabel("割合")
    ax.set_title(f"平均 {counts.mean():.3f} 本（= 20 × 0.05）")

    plots.save(fig, "fig-9-3-family-wise-error.png")


def main() -> None:
    plots.setup()
    with sim.Timer("9-3 20本の指標"):
        p = sim.repeat(one_trial, trials=TRIALS, seed=SEED, progress=False)

    counts = (p < ALPHA).sum(axis=1)
    # 先頭 m 本だけを見たときの FWER。同じ 10,000 回から m=1..20 が全部取れる。
    observed = np.array([(p[:, :m] < ALPHA).any(axis=1).mean() for m in range(1, M + 1)])
    theory = 1 - (1 - ALPHA) ** np.arange(1, M + 1)

    fwer = observed[-1]
    se = np.sqrt(fwer * (1 - fwer) / TRIALS)
    print(f"真の差はゼロの指標を {M} 本、n={N}/群、名目 α={ALPHA}、{TRIALS:,} 回\n")
    print(f"  少なくとも1本が有意   実測 {fwer:.4f} ± {1.96 * se:.4f}")
    print(f"                        理論 {theory[-1]:.4f}   （1 - 0.95^{M}）")
    print(f"                        差   {abs(fwer - theory[-1]):.4f}\n")
    print(f"  有意になった本数の平均 {counts.mean():.3f} 本（理論 {M * ALPHA:.3f} 本）")
    print(f"  1本も有意にならなかった割合 {(counts == 0).mean():.4f}")
    print(f"  3本以上が有意になった割合   {(counts >= 3).mean():.4f}\n")
    print(f"{'m':>4}{'実測':>10}{'理論':>10}")
    for m in (1, 3, 5, 10, 20):
        print(f"{m:>4}{observed[m - 1]:>10.4f}{theory[m - 1]:>10.4f}")
    print("\n「20本のうち1本が有意でした」は、報告としてほぼ情報を持たない。")
    draw(observed, counts)


if __name__ == "__main__":
    main()
