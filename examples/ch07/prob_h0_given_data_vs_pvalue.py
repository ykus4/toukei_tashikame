"""p値は Pr[データ | H₀] であって、Pr[H₀ | データ] ではない。

「p = 0.03 だから、帰無仮説が正しい確率は3%」は誤りである。条件が逆になっている。
第3-6節のベイズの定理がまさにこの向きの取り違えを扱っていて、ここではそれを
数え上げで見る。

帰無仮説が事前確率50%で真である世界を100,000回作り、p < 0.05 で棄却した回だけを
集めて、そのうち何割が本当は H₀ の世界だったかを数える。α=0.05 を守っていても、
棄却したうちの1割ちかくは的外れになる。α は「H₀ の世界で騒ぐ率」であって、
「騒いだときに間違っている率」ではない。

    uv run python examples/ch07/prob_h0_given_data_vs_pvalue.py
"""

import numpy as np
from scipy import optimize, stats

from toukei_tashikame import plots, power

N = 30            # 各群のサンプルサイズ
ALPHA = 0.05
PRIOR_H0 = 0.5    # H₀ が真である事前確率
WORLDS = 100_000


def main() -> None:
    plots.setup()

    # 検出力がちょうど 0.5 になる真の差を探す。「五分五分で見つかる」設計にそろえる。
    d_star = optimize.brentq(
        lambda d: power.power_ttest(N, d, ALPHA, kind="two-sample") - 0.5, 0.01, 2.0
    )
    pw = power.power_ttest(N, d_star, ALPHA, kind="two-sample")
    print(f"--- 世界の設計（各群 n={N}, α={ALPHA}）---")
    print(f"  H₀ が真である事前確率  {PRIOR_H0}")
    print(f"  H₁ のときの真の差      d = {d_star:.4f}（検出力がちょうど {pw:.4f} になる差）")

    # --- 100,000 個の世界を作る。半分では本当に差がなく、半分では差がある ---
    rng = np.random.default_rng(77)
    h0_true = rng.random(WORLDS) < PRIOR_H0
    delta = np.where(h0_true, 0.0, d_star)
    a = rng.normal(0.0, 1.0, size=(WORLDS, N))
    b = rng.normal(delta[:, None], 1.0, size=(WORLDS, N))

    # 等分散のt検定。100,000回ぶんをまとめて計算する
    sp2 = (a.var(axis=1, ddof=1) + b.var(axis=1, ddof=1)) / 2
    t = (b.mean(axis=1) - a.mean(axis=1)) / np.sqrt(sp2 * 2 / N)
    p = 2 * stats.t.sf(np.abs(t), df=2 * (N - 1))
    reject = p < ALPHA

    n_h0, n_h1 = int(h0_true.sum()), int((~h0_true).sum())
    tp = int((reject & ~h0_true).sum())     # 差があり、見つけた
    fp = int((reject & h0_true).sum())      # 差はないのに、騒いだ
    fn = n_h1 - tp
    tn = n_h0 - fp

    print(f"\n--- {WORLDS:,} 個の世界の内訳 ---")
    print("                    棄却した      棄却しなかった      計")
    print(f"  H₀ が真（差なし） {fp:>8,}      {tn:>10,}   {n_h0:>8,}")
    print(f"  H₁ が真（差あり） {tp:>8,}      {fn:>10,}   {n_h1:>8,}")
    print(f"  計                {fp + tp:>8,}      {tn + fn:>10,}   {WORLDS:>8,}")

    p_reject_given_h0 = fp / n_h0
    p_h0_given_reject = fp / (fp + tp)
    analytic = PRIOR_H0 * ALPHA / (PRIOR_H0 * ALPHA + (1 - PRIOR_H0) * pw)

    print("\n--- 2つの条件付き確率は別物 ---")
    print(f"  Pr[p<0.05 | H₀]   {p_reject_given_h0:.4f}"
          "   ← 検定が約束しているのはこちら（= α）")
    print(f"  Pr[H₀ | p<0.05]   {p_h0_given_reject:.4f}"
          "   ← 読者が知りたいのはこちら")
    print(f"  ベイズの定理で解くと {analytic:.4f}"
          "  （= 0.5×α ÷ (0.5×α + 0.5×検出力)）")
    print("\n  p < 0.05 で棄却したうち、およそ"
          f"{100 * p_h0_given_reject:.0f}%は「差が無い世界」から来ている。")
    print("  αを守ることと、棄却が当たっていることは別の話である")

    # --- 事前確率を振ると、同じ p<0.05 の意味が変わる ---
    print("\n--- 事前確率を変えると（検出力は同じ）---")
    print("   Pr[H₀]     Pr[H₀ | p<0.05]")
    for prior in (0.1, 0.5, 0.9, 0.99):
        v = prior * ALPHA / (prior * ALPHA + (1 - prior) * pw)
        print(f"    {prior:.2f}          {v:.4f}")
    print("  同じ「p = 0.04」でも、探している仮説がありそうかどうかで意味が変わる")

    # --- 図 ---
    fig, axes = plots.figure(1, 2)
    ax = axes[0]
    priors = np.linspace(0.001, 0.999, 400)
    curve = priors * ALPHA / (priors * ALPHA + (1 - priors) * pw)
    ax.plot(priors, curve, color=plots.PALETTE["posterior"], lw=1.4, zorder=4)
    ax.axhline(ALPHA, color=plots.PALETTE["truth"], lw=1.1, zorder=5)
    ax.annotate("α = 0.05", xy=(0.03, ALPHA + 0.02), fontsize=6.0,
                color=plots.PALETTE["truth"])
    ax.scatter([PRIOR_H0], [p_h0_given_reject], s=14, zorder=6,
               color=plots.PALETTE["estimate"])
    ax.annotate(f"数え上げ {p_h0_given_reject:.4f}", xy=(PRIOR_H0, p_h0_given_reject),
                xytext=(6, 10), textcoords="offset points", fontsize=6.0,
                color=plots.PALETTE["estimate"])
    ax.set_xlabel("H₀ が真である事前確率")
    ax.set_ylabel("Pr[H₀ | p<0.05]")
    ax.set_title("棄却したときに間違っている確率")

    ax = axes[1]
    bars = [n_h0 / WORLDS, n_h1 / WORLDS, fp / (fp + tp), tp / (fp + tp)]
    ax.bar([0, 1], bars[:2], color=[plots.PALETTE["null"], plots.PALETTE["alt"]], width=0.7)
    ax.bar([3, 4], bars[2:], color=[plots.PALETTE["null"], plots.PALETTE["alt"]], width=0.7)
    ax.set_xticks([0, 1, 3, 4])
    ax.set_xticklabels(["H₀", "H₁", "H₀", "H₁"])
    ax.annotate("全部の世界", xy=(0.5, -0.22), xycoords=("data", "axes fraction"),
                ha="center", fontsize=6.2, color=plots.PALETTE["ink2"])
    ax.annotate("p<0.05 だけ", xy=(3.5, -0.22), xycoords=("data", "axes fraction"),
                ha="center", fontsize=6.2, color=plots.PALETTE["ink2"])
    for x, v in zip([0, 1, 3, 4], bars, strict=True):
        ax.annotate(f"{v:.3f}", xy=(x, v), xytext=(0, 2), textcoords="offset points",
                    ha="center", fontsize=6.0, color=plots.PALETTE["ink2"])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("割合")
    ax.set_title("棄却で絞ると内訳が変わる")
    fig.tight_layout()
    plots.save(fig, "fig-7-7-inverse-probability.png")


if __name__ == "__main__":
    main()
