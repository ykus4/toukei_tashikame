"""感度99%の検査で陽性。それでも病気である確率は2%しかない。10,000人で数える。

ベイズの式を暗記しても直感は動かない。動くのは人数で数えたときである。有病率0.1%の
集団 10,000 人を検査すると、病気の人はおよそ 10 人。そのうち 99% が陽性になるので
真陽性はおよそ 9.9 人。一方、健康な 9,990 人のうち 5% が誤って陽性になるので偽陽性は
およそ 499.5 人。陽性の 509 人のうち本当に病気なのは 9.9 人 — これが 2% の正体である。

効いているのは感度でも特異度でもなく、**事前確率（有病率）の低さ**。ここを見誤ると、
検査結果という「証拠」の重みを何倍にも読み違える。

    uv run python examples/ch03/bayes_medical_test_10000_people.py
"""

import numpy as np

from toukei_tashikame import plots

PREVALENCE = 0.001      # 有病率 0.1%
SENSITIVITY = 0.99      # 感度: 病気の人が陽性になる確率
SPECIFICITY = 0.95      # 特異度: 健康な人が陰性になる確率
N_PEOPLE = 10_000
N_SIMS = 10_000
SEED = 8


def ppv_formula() -> float:
    """ベイズの式による陽性的中率 Pr[病気 | 陽性]。"""
    num = PREVALENCE * SENSITIVITY
    den = num + (1 - PREVALENCE) * (1 - SPECIFICITY)
    return num / den


def main() -> None:
    plots.setup()

    rng = np.random.default_rng(SEED)
    # 1人ずつ回さず、人数を二項分布から引く。1回のシミュレーションは
    # 「10,000人のうち何人が病気で、そのうち何人が陽性になったか」の3つの数で足りる。
    sick = rng.binomial(N_PEOPLE, PREVALENCE, size=N_SIMS)
    tp = rng.binomial(sick, SENSITIVITY)                       # 真陽性
    fp = rng.binomial(N_PEOPLE - sick, 1 - SPECIFICITY)        # 偽陽性
    fn = sick - tp                                             # 偽陰性
    tn = (N_PEOPLE - sick) - fp                                # 真陰性
    positive = tp + fp

    print(f"--- 10,000人を検査、を {N_SIMS:,} 回（seed={SEED}）。人数の平均 ---")
    print(f"  {'':<10}{'陽性':>10}{'陰性':>10}{'計':>10}")
    print(f"  {'病気':<10}{tp.mean():>10.1f}{fn.mean():>10.1f}{sick.mean():>10.1f}")
    print(f"  {'健康':<10}{fp.mean():>10.1f}{tn.mean():>10.1f}{(N_PEOPLE - sick).mean():>10.1f}")
    print(f"  {'計':<10}{positive.mean():>10.1f}{(fn + tn).mean():>10.1f}{N_PEOPLE:>10.1f}")

    ppv_sim = float(tp.sum() / positive.sum())
    ppv_theory = ppv_formula()
    per_sim = tp / np.maximum(positive, 1)
    print(f"\n  陽性的中率 PPV（全 {N_SIMS:,} 回をまとめて数える） {ppv_sim:.4f}")
    print(f"  ベイズの式  {PREVALENCE}×{SENSITIVITY} / "
          f"({PREVALENCE}×{SENSITIVITY} + {1 - PREVALENCE:.3f}×{1 - SPECIFICITY:.2f}) "
          f"= {ppv_theory:.4f}")
    print(f"  差 {abs(ppv_sim - ppv_theory):.4f}")
    print(f"  1回ごとの PPV の散らばり: 中央値 {np.median(per_sim):.4f}、"
          f"5〜95%点 {np.quantile(per_sim, 0.05):.4f}〜{np.quantile(per_sim, 0.95):.4f}")

    print("\n--- 陽性だと言われたら、確率はどう動いたか ---")
    print(f"  検査前  {PREVALENCE:.4f}（1000人に1人）")
    print(f"  検査後  {ppv_theory:.4f}（{1 / ppv_theory:.0f}人に1人）")
    print(f"  → {ppv_theory / PREVALENCE:.1f} 倍に上がった。上がってはいるが、"
          "まだ 98% は病気ではない")

    print("\n--- 有病率だけを変えると（感度99%・特異度95%は据え置き）---")
    print(f"  {'有病率':>8}  {'PPV':>8}")
    for prev in [0.001, 0.01, 0.05, 0.10, 0.30, 0.50]:
        p = prev * SENSITIVITY / (prev * SENSITIVITY + (1 - prev) * (1 - SPECIFICITY))
        print(f"  {prev:>8.3f}  {p:>8.4f}")
    print("  検査の性能は1文字も変わっていない。動いたのは事前確率だけ")

    fig, (ax1, ax2) = plots.figure(1, 2, w=2.0, h=1.0)

    # 左: 10,000人が4つの箱に分かれる樹形図。
    ax1.axis("off")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    nodes = [
        (0.02, 0.52, "10,000 人", plots.PALETTE["ink"]),
        (0.36, 0.86, f"病気 {sick.mean():.1f} 人", plots.PALETTE["truth"]),
        (0.36, 0.20, f"健康 {(N_PEOPLE - sick).mean():.1f} 人", plots.PALETTE["ink2"]),
        (0.70, 0.96, f"陽性 {tp.mean():.1f}", plots.PALETTE["reject"]),
        (0.70, 0.76, f"陰性 {fn.mean():.1f}", plots.PALETTE["ink2"]),
        (0.70, 0.32, f"陽性 {fp.mean():.1f}", plots.PALETTE["reject"]),
        (0.70, 0.06, f"陰性 {tn.mean():.1f}", plots.PALETTE["ink2"]),
    ]
    for x, y, text, color in nodes:
        ax1.text(x, y, text, fontsize=6.4, color=color, va="center", ha="left")
    edges = [((0.16, 0.52), (0.34, 0.86), f"{PREVALENCE:.3f}"),
             ((0.16, 0.52), (0.34, 0.20), f"{1 - PREVALENCE:.3f}"),
             ((0.60, 0.86), (0.68, 0.96), f"{SENSITIVITY:.2f}"),
             ((0.60, 0.86), (0.68, 0.76), f"{1 - SENSITIVITY:.2f}"),
             ((0.60, 0.20), (0.68, 0.32), f"{1 - SPECIFICITY:.2f}"),
             ((0.60, 0.20), (0.68, 0.06), f"{SPECIFICITY:.2f}")]
    for (x0, y0), (x1, y1), label in edges:
        ax1.plot([x0, x1], [y0, y1], color=plots.PALETTE["grid"], lw=0.8, zorder=1)
        ax1.text((x0 + x1) / 2, (y0 + y1) / 2, label, fontsize=5.6,
                 color=plots.PALETTE["ink2"], ha="center", va="bottom")
    ax1.set_title("10,000人がどこへ行くか")

    # 右: 陽性者の内訳。ほとんどが偽陽性であることを面積で見せる。
    ax2.barh([0], [tp.mean()], color=plots.PALETTE["truth"], height=0.55, lw=0)
    ax2.barh([0], [fp.mean()], left=[tp.mean()], color=plots.PALETTE["reject"],
             alpha=0.75, height=0.55, lw=0)
    ax2.set_yticks([0])
    ax2.set_yticklabels([f"陽性 {positive.mean():.0f} 人"])
    ax2.set_xlabel("人数")
    ax2.annotate(f"真陽性 {tp.mean():.1f} 人", xy=(tp.mean(), 0.32),
                 fontsize=6.2, color=plots.PALETTE["truth"], ha="left")
    ax2.annotate(f"偽陽性 {fp.mean():.1f} 人", xy=(tp.mean() + fp.mean() / 2, -0.34),
                 fontsize=6.2, color=plots.PALETTE["reject"], ha="center", va="top")
    ax2.set_ylim(-0.8, 0.8)
    ax2.set_title(f"PPV = {ppv_sim:.4f}")

    fig.tight_layout()
    plots.save(fig, "fig-3-6-ppv-tree.png")


if __name__ == "__main__":
    main()
