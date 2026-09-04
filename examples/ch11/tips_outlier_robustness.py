"""外れ値1点で相関はどこまで動くか — tips に 500 ドルのチップを1件足す。

実データ（seaborn の tips、n=244）で total_bill と tip の相関を測ると 0.68 前後。
そこへ「10 ドルの食事に 500 ドルのチップ」という 1 件だけを足す。245 件のうちの 1 件、
0.4% である。それだけで Pearson は 0.68 から 0 のあたりまで落ち、符号も定まらなくなる。
順位しか見ない Spearman と Kendall はほとんど動かない。

Pearson は偏差の積を足し上げるので、平均から大きく離れた 1 点が和のほとんどを占めて
しまう。順位に直すと 500 ドルも 10.1 ドルも「245 位」でしかない。頑健とはこの意味で、
情報を捨てることで得られている性質である。捨ててよいかどうかは、その外れ値が測定
ミスなのか本物の稀な事象なのかで決まる。

    uv run python examples/ch11/tips_outlier_robustness.py
"""

import numpy as np
import pandas as pd
from scipy import stats

from toukei_tashikame import datasets, plots

OUTLIER_BILL, OUTLIER_TIP = 10.0, 500.0


def three_correlations(x, y) -> dict[str, float]:
    """Pearson / Spearman / Kendall を同じ2列に対してまとめて取る。"""
    return {
        "Pearson r": float(stats.pearsonr(x, y).statistic),
        "Spearman ρ": float(stats.spearmanr(x, y).statistic),
        "Kendall τ": float(stats.kendalltau(x, y).statistic),
    }


def main() -> None:
    plots.setup()
    tips = datasets.tips()
    x, y = tips["total_bill"].to_numpy(), tips["tip"].to_numpy()

    print(f"--- 11-3 tips（seaborn 同梱, n={len(tips)}）total_bill と tip ---")
    base = three_correlations(x, y)
    print(f"  tip の最大値 {y.max():.2f} ドル、中央値 {np.median(y):.2f} ドル")
    for k, v in base.items():
        print(f"  {k:<12} {v:.4f}")

    x2 = np.append(x, OUTLIER_BILL)
    y2 = np.append(y, OUTLIER_TIP)
    after = three_correlations(x2, y2)
    print(f"\n--- 外れ値を1点だけ足す（total_bill={OUTLIER_BILL:.0f}, "
          f"tip={OUTLIER_TIP:.0f}）: n={len(y2)} の 1 件、{100 / len(y2):.1f}% ---")
    print("  係数           もと      追加後     変化")
    for k in base:
        print(f"  {k:<12} {base[k]: .4f}   {after[k]: .4f}   {after[k] - base[k]:+.4f}")
    print(f"  ← Pearson は {base['Pearson r']:.4f} から {after['Pearson r']:.4f} まで落ちて"
          "符号すら定まらない。")
    print(f"    Spearman の変化は {abs(after['Spearman ρ'] - base['Spearman ρ']):.4f}、"
          f"Kendall は {abs(after['Kendall τ'] - base['Kendall τ']):.4f} にとどまる")

    # 外れ値の大きさを振る。どこから壊れるのかを見ておく。
    sizes = np.array([5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0])
    rows = []
    for t in sizes:
        rows.append({"tip": t, **three_correlations(np.append(x, OUTLIER_BILL), np.append(y, t))})
    sweep = pd.DataFrame(rows).set_index("tip")
    print("\n--- 外れ値のチップ額を振る（total_bill は 10 ドルのまま）---")
    print(sweep.to_string(float_format=lambda v: f"{v: .4f}"))
    print("  ← 100 ドルを超えたあたりから Pearson が崩れる。順位ベースの2つは"
          "「一番上の1つ」でしかないので、額をいくら増やしても頭打ちになる")

    print(f"\n  順位に直せば安全、という話ではない。Pearson が失った {abs(after['Pearson r'] - base['Pearson r']):.4f} は")
    print("  雑音ではなく「この1件は本当にあった」という情報でもある。捨てる前に、なぜ外れたかを見る")

    # --- 図 ---
    fig, axes = plots.figure(1, 3, w=1.95, h=0.95)
    for ax, (xx, yy, title) in zip(
        axes,
        [(x, y, f"もとの {len(y)} 件"), (x2, y2, f"外れ値を1点足した {len(y2)} 件")],
        strict=False,
    ):
        ax.scatter(xx, yy, s=6, color=plots.PALETTE["data"], lw=0, alpha=0.7, zorder=3)
        b, a = np.polyfit(xx, yy, 1)
        xs = np.linspace(0, 55, 50)
        ax.plot(xs, a + b * xs, color=plots.PALETTE["estimate"], lw=1.2, zorder=4)
        r = stats.pearsonr(xx, yy).statistic
        rho = stats.spearmanr(xx, yy).statistic
        ax.set_title(f"{title}\nr = {r:.4f} / ρ = {rho:.4f}")
        ax.set_xlabel("会計額 (ドル)")
    axes[0].set_ylabel("チップ (ドル)")
    axes[1].scatter([OUTLIER_BILL], [OUTLIER_TIP], s=18, color=plots.PALETTE["reject"], zorder=5)
    axes[1].annotate("この1点", xy=(OUTLIER_BILL, OUTLIER_TIP), xytext=(6, -6),
                     textcoords="offset points", fontsize=6.0, color=plots.PALETTE["reject"])

    ax = axes[2]
    for key, style in [("Pearson r", "-"), ("Spearman ρ", "--"), ("Kendall τ", ":")]:
        ax.plot(sweep.index, sweep[key], style, lw=1.2, color=plots.PALETTE["estimate"], zorder=3)
        ax.annotate(key, xy=(sweep.index[-1], sweep[key].iloc[-1]), xytext=(-2, 3),
                    textcoords="offset points", ha="right", fontsize=6.0,
                    color=plots.PALETTE["estimate"])
    ax.set_xscale("log")
    ax.set_xlabel("足した1点のチップ額 (ドル)")
    ax.set_ylabel("相関係数")
    ax.set_title("外れ値の大きさへの反応")
    fig.tight_layout()
    plots.save(fig, "fig-11-3-tips-outlier-robustness.png")


if __name__ == "__main__":
    main()
