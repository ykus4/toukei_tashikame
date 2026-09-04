"""検出力を数え上げる — 真の効果 d=0.5、n=30/群で、何回に何回棄却できるか。

検出力は「対立仮説が本当なら、どれくらいの割合で棄却できるか」でしかない。第7章で
第一種の誤りを数えたのとまったく同じ手続きで、変えるのはデータの生成側だけ——群Bの
平均を 0 から d にずらす。それだけで、同じコードが α を数える装置から検出力を数える
装置に変わる。

数えた値を statsmodels の解析解と突き合わせる。式が合うことより、**式が何を数えて
いるのかが見えること**のほうが大事なので、先に数えてから式を当てる。

    uv run python examples/ch10/power_by_counting.py
"""

import numpy as np
from scipy import stats
from statsmodels.stats.power import TTestIndPower

from toukei_tashikame import plots, power

D, N, ALPHA, TRIALS, SEED = 0.5, 30, 0.05, 10_000, 101


def main() -> None:
    plots.setup()

    # --- 1. 数え上げ。10,000回検定して、p < α だった割合を数えるだけ ---
    res = power.power_sim(N, D, alpha=ALPHA, trials=TRIALS, seed=SEED)

    # --- 2. 解析解。非心 t 分布の裾（本書の power_ttest）と statsmodels ---
    theory = power.power_ttest(N, D, alpha=ALPHA)
    sm = float(TTestIndPower().power(effect_size=D, nobs1=N, alpha=ALPHA,
                                     ratio=1.0, alternative="two-sided"))

    print(f"--- 真の効果 d={D}、n={N}/群、α={ALPHA} で {TRIALS:,} 回検定した ---")
    print(f"  シミュレーション（数え上げ）  {res.rate:.4f} ± {1.96 * res.se:.4f}")
    print(f"  power.power_ttest（非心t）    {theory:.4f}")
    print(f"  statsmodels TTestIndPower     {sm:.4f}")
    print(f"  数え上げ − statsmodels        {res.rate - sm:+.4f}"
          f"（数え直しの誤差 ±{1.96 * res.se:.4f} の内側）")
    print(f"\n  つまり {TRIALS:,} 回のうち {int(res.rate * TRIALS):,} 回は棄却できたが、"
          f"残り {int((1 - res.rate) * TRIALS):,} 回は")
    print(f"  効果が本当にあるのに見逃した。第二種の誤り β = {1 - res.rate:.4f}。")
    print("  効果があるのに半分以上は気づけない設計、というのがこの n の実態である。")

    # --- 3. 図。帰無分布と対立分布と棄却域を1枚に重ねる ---
    fig, axes = plots.figure(1, 2, w=1.6)

    df = 2 * (N - 1)
    crit = stats.t.ppf(1 - ALPHA / 2, df)
    ncp = D * np.sqrt(N / 2)
    x = np.linspace(-4.5, 7.5, 600)
    ax = axes[0]
    plots.null_vs_alt(ax, x, stats.t.pdf(x, df), x, stats.nct.pdf(x, df, ncp), crit=crit)
    ax.annotate(f"帰無 t({df})", xy=(-2.6, stats.t.pdf(-2.6, df) + 0.02),
                fontsize=6.0, color=plots.PALETTE["null"])
    ax.annotate(f"対立（非心度 {ncp:.2f}）", xy=(3.3, stats.nct.pdf(3.3, df, ncp) + 0.02),
                fontsize=6.0, color=plots.PALETTE["alt"])
    ax.annotate(f"棄却域の面積 = 検出力 {theory:.3f}", xy=(crit + 0.15, 0.30),
                fontsize=6.0, color=plots.PALETTE["reject"])
    ax.set_xlabel("t 統計量")
    ax.set_ylabel("密度")
    ax.set_title(f"d={D}, n={N}/群 の検出力")

    ax = axes[1]
    t_sim = stats.t.isf(res.pvalues / 2, df)   # p値をt統計量に戻す（両側なので絶対値）
    ax.hist(t_sim, bins=50, density=True, color=plots.PALETTE["data"], alpha=0.55, lw=0)
    xp = x[x > 0]
    # |t| の密度は折り返し。負側の裾を足す（2倍ではない）
    ax.plot(xp, stats.nct.pdf(xp, df, ncp) + stats.nct.pdf(-xp, df, ncp),
            color=plots.PALETTE["truth"], lw=1.2, ls="--", dashes=(4, 2.0), zorder=5)
    ax.axvline(crit, color=plots.PALETTE["reject"], lw=0.9, ls="--", dashes=(4, 2.2))
    ax.annotate(f"|t| > {crit:.2f} が {res.rate:.1%}", xy=(0.97, 0.88),
                xycoords="axes fraction", ha="right", fontsize=6.0,
                color=plots.PALETTE["reject"])
    ax.set_xlabel("|t|（10,000回ぶん）")
    ax.set_ylabel("密度")
    ax.set_title("数え上げた |t| の分布")

    fig.tight_layout()
    plots.save(fig, "fig-10-1-power-by-counting.png")


if __name__ == "__main__":
    main()
