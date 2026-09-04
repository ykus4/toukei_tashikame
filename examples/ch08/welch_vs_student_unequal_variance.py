"""崩し — 等分散を仮定した Student の t 検定を、不均衡・不等分散のデータに当てる。

標本サイズが揃っていれば Student は分散比の違いにかなり頑健である。壊れるのは
**サイズと分散が食い違うとき**で、しかも壊れ方に向きがある。小さい群のほうが大きな
分散を持つと、プールした分散が真の標準誤差を過小に見積もり、第一種の誤りが名目の5%を
大きく超える。逆向き（小さい群が小さな分散）だと今度は保守的になりすぎ、検出力を捨てる。

どちらの向きでも Welch は5%の近くにとどまる。本書が2標本t検定の既定を Welch に置く
理由がここにある。

    uv run python examples/ch08/welch_vs_student_unequal_variance.py
"""

import numpy as np
from scipy import stats

from toukei_tashikame import plots

TRIALS, SEED, ALPHA = 10_000, 83, 0.05
N1, N2 = 10, 40

# (ラベル, 小さい群 n=10 の σ, 大きい群 n=40 の σ, 図の見出し)
DESIGNS = [
    ("n=10,σ=4 / n=40,σ=1", 4.0, 1.0, "小群が\n大分散"),
    ("n=10,σ=1 / n=40,σ=4", 1.0, 4.0, "小群が\n小分散"),
    ("n=10,σ=1 / n=40,σ=1", 1.0, 1.0, "等分散\n（対照）"),
]


def type1_errors(sd1: float, sd2: float, seed: int) -> tuple[float, float]:
    """帰無（両群とも平均0）のデータを TRIALS 組作り、Student と Welch の棄却率を返す。

    試行ごとにループを回すかわりに ``(TRIALS, n)`` の行列を1度に引き、``axis=1`` で
    まとめて検定する。数えているものは1本ずつ回すのと同じで、時間だけが短くなる。
    """
    rng = np.random.default_rng(seed)
    a = rng.normal(0.0, sd1, size=(TRIALS, N1))
    b = rng.normal(0.0, sd2, size=(TRIALS, N2))
    p_student = stats.ttest_ind(a, b, axis=1, equal_var=True).pvalue
    p_welch = stats.ttest_ind(a, b, axis=1, equal_var=False).pvalue
    return float((p_student < ALPHA).mean()), float((p_welch < ALPHA).mean())


def se(rate: float) -> float:
    """数え上げそのものの標準誤差。この幅より小さい差は数え直すだけで動く。"""
    return float(np.sqrt(rate * (1 - rate) / TRIALS))


def main() -> None:
    plots.setup()
    print(f"帰無が真（両群とも平均0）のデータを {TRIALS:,} 組。名目 α = {ALPHA}")
    print(f"\n{'設計':<19}   {'Student':^17}   {'Welch':^17}")

    rows = []
    for i, (label, sd1, sd2, tick) in enumerate(DESIGNS):
        s, w = type1_errors(sd1, sd2, SEED + 100 * i)
        rows.append((label, s, w, tick))
        print(f"{label:<21}  {s:.4f} ± {1.96 * se(s):.4f}   {w:.4f} ± {1.96 * se(w):.4f}")

    s_bad, s_cons = rows[0][1], rows[1][1]
    print(f"\n  設計値5%が {s_bad:.4f}（{s_bad / ALPHA:.1f}倍）まで膨らむのは、"
          "小さい群のほうが分散が大きいとき。")
    print(f"  向きが逆だと今度は {s_cons:.4f} まで潰れる。過大にも過小にもなる、"
          "というのが等分散仮定の破れ方である。")
    print("  Welch はどちらの向きでも 5% の近くにとどまる。")

    # --- 図 ---
    fig, ax = plots.figure(w=1.3)
    x = np.arange(len(rows))
    width = 0.36
    student = [r[1] for r in rows]
    welch = [r[2] for r in rows]
    ax.bar(x - width / 2, student, width, color=plots.PALETTE["reject"], alpha=0.85,
           lw=0, label="Student（等分散を仮定）")
    ax.bar(x + width / 2, welch, width, color=plots.PALETTE["estimate"], alpha=0.85,
           lw=0, label="Welch")
    for xi, (s, w) in enumerate(zip(student, welch, strict=True)):
        ax.annotate(f"{s:.4f}", xy=(xi - width / 2, s), xytext=(0, 2),
                    textcoords="offset points", ha="center", fontsize=6.0,
                    color=plots.PALETTE["reject"])
        ax.annotate(f"{w:.4f}", xy=(xi + width / 2, w), xytext=(0, 2),
                    textcoords="offset points", ha="center", fontsize=6.0,
                    color=plots.PALETTE["estimate"])
    plots.mark_truth(ax, ALPHA, "名目 α = 0.05", axis="y")
    ax.set_xticks(x)
    ax.set_xticklabels([r[3] for r in rows])
    ax.set_ylabel("第一種の誤り")
    ax.set_title(f"不均衡（n=10 vs 40）での第一種の誤り（{TRIALS:,}回）")
    ax.legend(loc="upper right")
    fig.tight_layout()
    plots.save(fig, "fig-8-3-unequal-variance-type1.png")


if __name__ == "__main__":
    main()
