"""条件付き確率とは、行を絞ってから数え直すこと。それだけ。

$\\Pr[A \\mid B] = \\Pr[A \\cap B] / \\Pr[B]$ という式は、割り算の意味を掴むまで
何も語らない。サイコロ2個の表を10万行つくり、「1個目が5以上」の行だけを残して
同じ数え上げをもう一度やると、式の分母がしていることが目で見える。分母 $\\Pr[B]$ は
**残った行数で割り直す**という操作そのものである。

pandas の `df[df.d1 >= 5]` が条件付き確率だと分かれば、この節は終わりでよい。

    uv run python examples/ch03/conditional_probability_by_filtering.py
"""

import numpy as np

from toukei_tashikame import plots

N_ROLLS = 100_000
SEED = 7


def main() -> None:
    plots.setup()

    rng = np.random.default_rng(SEED)
    d1 = rng.integers(1, 7, size=N_ROLLS)
    d2 = rng.integers(1, 7, size=N_ROLLS)
    total = d1 + d2

    a = total >= 9            # 事象 A: 合計が9以上
    b = d1 >= 5               # 事象 B: 1個目が5以上

    print(f"--- 2個のサイコロを {N_ROLLS:,} 回（seed={SEED}）---")
    print(f"  Pr[A: 合計>=9]              {a.mean():.4f}   （理論 {10 / 36:.4f} = 10/36）")
    print(f"  Pr[B: 1個目>=5]             {b.mean():.4f}   （理論 {12 / 36:.4f} = 12/36）")
    print(f"  Pr[A かつ B]                {(a & b).mean():.4f}   （理論 {7 / 36:.4f} =  7/36）")

    print("\n--- やり方1: 式で割る ---")
    by_formula = (a & b).mean() / b.mean()
    print(f"  Pr[A かつ B] / Pr[B] = {(a & b).mean():.4f} / {b.mean():.4f} = {by_formula:.4f}")

    print("\n--- やり方2: 行を絞ってから数え直す ---")
    kept = total[b]           # 1個目が5以上だった行だけを残す
    by_filter = (kept >= 9).mean()
    print(f"  残った行  {kept.size:,} 行 / {N_ROLLS:,} 行")
    print(f"  その中で合計>=9  {int((kept >= 9).sum()):,} 行 → {by_filter:.4f}")
    print(f"  理論 7/12 = {7 / 12:.4f}")
    print(f"\n  2つのやり方の差 {abs(by_formula - by_filter):.2e} ← 同じ計算をしている")
    print(f"  条件をつけると 0.28 → {by_filter:.2f}。"
          "「1個目が5以上」という情報が、合計の分布そのものを右に押した")

    print("\n--- 条件を変えると分母が変わる ---")
    for name, cond in [("1個目>=5", d1 >= 5), ("1個目==6", d1 == 6),
                       ("1個目<=2", d1 <= 2), ("条件なし", np.ones(N_ROLLS, bool))]:
        sub = total[cond]
        print(f"  {name:<10} 残る行 {sub.size:>6,} 行   Pr[合計>=9 | 条件] = {(sub >= 9).mean():.4f}")

    fig, (ax1, ax2) = plots.figure(1, 2, w=2.0, h=0.95, sharey=True)
    bins = np.arange(1.5, 13.5)

    for ax, mask, title in [
        (ax1, np.ones(N_ROLLS, bool), f"全 {N_ROLLS:,} 行"),
        (ax2, b, f"1個目>=5 の {int(b.sum()):,} 行だけ"),
    ]:
        sub = total[mask]
        counts, edges = np.histogram(sub, bins=bins)
        prob = counts / sub.size
        centers = edges[:-1] + 0.5
        hit = centers >= 9
        ax.bar(centers[~hit], prob[~hit], width=0.85, color=plots.PALETTE["data"],
               alpha=0.65, lw=0)
        ax.bar(centers[hit], prob[hit], width=0.85, color=plots.PALETTE["reject"],
               alpha=0.85, lw=0)
        ax.set_title(title)
        ax.set_xlabel("2個の合計")
        ax.set_xticks(range(2, 13, 2))
        ax.annotate(f"合計>=9 は {prob[hit].sum():.4f}",
                    xy=(0.03, 0.95), xycoords="axes fraction", ha="left", va="top",
                    fontsize=6.2, color=plots.PALETTE["reject"])
        ax.set_ylim(0, 0.20)
    ax1.set_ylabel("相対頻度")

    fig.tight_layout()
    plots.save(fig, "fig-3-4-conditioning-is-filtering.png")


if __name__ == "__main__":
    main()
