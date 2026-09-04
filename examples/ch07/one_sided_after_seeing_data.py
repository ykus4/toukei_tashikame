"""崩し — データを見てから片側検定に切り替えると、αは倍になる。

片側検定そのものは正しい道具である。壊れるのは順番で、「差がプラス側に出ていたから
プラス側の片側検定にした」とやった瞬間、棄却域はデータに合わせて動く可動式になる。
実際に切り取っているのは両側の 5% + 5% = 10% で、名目の 5% ではない。

差がまったく無いデータを10,000回作り、(1) 事前に決めた両側検定、(2) 事前に決めた
片側検定、(3) 符号を見てから向きを決める片側検定、の3つで第一種の誤りを数える。
壊れるのは (3) だけである。

    uv run python examples/ch07/one_sided_after_seeing_data.py
"""

import numpy as np
from scipy import stats

from toukei_tashikame import plots, sim, testing

N = 40
ALPHA = 0.05
TRIALS = 10_000
DF = N - 1


def null_t(rng) -> float:
    """真の効果がゼロ（μ=0）の標本から、1標本t検定の統計量を返す。"""
    x = rng.normal(0.0, 1.0, size=N)
    return testing.t_1samp(x, 0.0).stat


def main() -> None:
    plots.setup()

    t = sim.repeat(null_t, trials=TRIALS, seed=78, progress=False)

    # (1) 事前に決めた両側検定
    p_two = 2 * stats.t.sf(np.abs(t), df=DF)
    # (2) 事前に「プラス側だけを見る」と決めた片側検定
    p_pre = stats.t.sf(t, df=DF)
    # (3) 符号を見てから、都合のよい側に切り替える片側検定
    p_post = np.where(t > 0, stats.t.sf(t, df=DF), stats.t.cdf(t, df=DF))

    def report(label: str, p) -> float:
        rate = float((p < ALPHA).mean())
        se = np.sqrt(rate * (1 - rate) / TRIALS)
        print(f"  {label:<28} {rate:.4f} ± {1.96 * se:.4f}   "
              f"（{int((p < ALPHA).sum()):,} / {TRIALS:,}）")
        return rate

    print(f"--- 真の効果ゼロのデータを {TRIALS:,} 回（n={N}, α={ALPHA}, seed=78）---")
    rate_two = report("(1) 事前に決めた両側", p_two)
    report("(2) 事前に決めた片側", p_pre)
    rate_post = report("(3) 符号を見てから片側", p_post)

    print(f"\n  (3) は (1) の {rate_post / rate_two:.2f} 倍。α=0.05 と書いてあるのに、"
          f"実際には {rate_post:.1%} で騒いでいる")
    print("  片側検定が悪いのではない。(2) はきちんと 0.05 に収まっている。"
          "壊すのは「見てから決めた」という順番")
    print(f"\n  (3) のp値は 0 から 0.5 の一様分布になる（p値の最大 "
          f"{p_post.max():.4f}）。")
    print("  1より大きい値が出ないよう半分に折りたたんだ結果で、"
          "下から5%を切ると本当は10%が切れる")

    # --- 図: 2つの手続きのp値の分布 ---
    fig, axes = plots.figure(1, 2)
    for ax, p, title, dens in (
        (axes[0], p_two, f"(1) 事前に決めた両側  {rate_two:.4f}", 1.0),
        (axes[1], p_post, f"(3) 符号を見てから片側  {rate_post:.4f}", 2.0),
    ):
        ax.hist(p, bins=20, range=(0, 1), density=True, color=plots.PALETTE["data"],
                alpha=0.55, lw=0)
        ax.axvspan(0, ALPHA, color=plots.PALETTE["reject"], alpha=0.55, lw=0, zorder=2)
        ax.axhline(dens, color=plots.PALETTE["truth"], lw=1.1, zorder=5)
        ax.annotate(f"密度 {dens:g}", xy=(0.55, dens), xytext=(0, 2),
                    textcoords="offset points", fontsize=6.0,
                    color=plots.PALETTE["truth"])
        ax.set_ylim(0, 2.5)
        ax.set_xlabel("p値")
        ax.set_ylabel("密度")
        ax.set_title(title)
    axes[1].annotate("同じ幅で切っても\n面積が倍", xy=(ALPHA, 2.05), xytext=(10, 0),
                     textcoords="offset points", fontsize=6.0,
                     color=plots.PALETTE["reject"])
    fig.tight_layout()
    plots.save(fig, "fig-7-8-post-hoc-one-sided.png")


if __name__ == "__main__":
    main()
