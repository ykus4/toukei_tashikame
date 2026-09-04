"""二元配置分散分析 — 交互作用があると「主効果」はもう単独では読めない。

要因が2つあるとき、分散分析は変動を「要因Aの効果」「要因Bの効果」「A×Bの交互作用」
「誤差」に分ける。交互作用とは「Aの効き方がBの水準によって変わる」こと。これが有意なら、
「Aの主効果は…」という言い方は平均を取ったぶんの話でしかなく、単独では意味が薄くなる。

データは Moore & Krantz (1980) の同調実験（``carData::Moore``, n=45）。被験者の権威主義
傾向 fcategory（low/medium/high）と、相手の地位 partner.status（low/high）で、同調行動の
回数 conformity を説明する。交互作用プロットの2本の線が平行でないことが、そのまま
交互作用の正体である。

平方和は Type II で取る。非釣り合い型（セルの人数が揃わない）データでは Type I が
式に書いた順番に依存してしまうためで、この45人は 2×3 のセルが揃っていない。

    uv run python examples/ch14/two_way_anova_interaction_moore.py
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

from toukei_tashikame import plots

# carData::Moore（n=45）。ダウンロードなしで動くよう45行をここに直接書いてある。
CONFORMITY = [8, 4, 8, 7, 10, 6, 12, 4, 13, 12, 4, 13, 7, 9, 9, 24, 6, 7, 23, 13, 8, 12,
              19, 12, 21, 9, 23, 7, 17, 14, 11, 16, 15, 20, 8, 12, 14, 14, 17, 7, 17, 13,
              16, 10, 15]
FCATEGORY = ["low", "high", "high", "low", "low", "low", "medium", "medium", "low", "low",
             "medium", "high", "low", "medium", "high", "high", "low", "high", "high",
             "high", "low", "low", "high", "medium", "low", "high", "low", "high",
             "medium", "medium", "high", "medium", "low", "medium", "medium", "low",
             "high", "medium", "medium", "medium", "medium", "high", "low", "high",
             "medium"]
PARTNER = ["low"] * 22 + ["high"] * 23

F_LEVELS = ["low", "medium", "high"]
P_LEVELS = ["low", "high"]


def main() -> None:
    plots.setup()
    df = pd.DataFrame({"conformity": CONFORMITY, "fcategory": FCATEGORY,
                       "partner": PARTNER})

    print(f"--- 14-5 Moore の同調実験（n={len(df)}）二元配置分散分析 ---")
    cells = df.pivot_table(index="fcategory", columns="partner", values="conformity",
                           aggfunc=["count", "mean"]).round(3)
    print("  セルごとの人数と平均")
    print("  " + cells.reindex(F_LEVELS).to_string().replace("\n", "\n  "))
    print("  ← セルの人数が揃っていない（非釣り合い型）ので Type II 平方和を使う")

    full = smf.ols("conformity ~ C(fcategory) * C(partner)", data=df).fit()
    add = smf.ols("conformity ~ C(fcategory) + C(partner)", data=df).fit()

    print("\n  交互作用ありのモデル（Type II 平方和）")
    tbl2 = anova_lm(full, typ=2)
    print("  " + tbl2.round(4).to_string().replace("\n", "\n  "))

    print("\n  交互作用なし（主効果だけ）のモデル")
    print("  " + anova_lm(add, typ=2).round(4).to_string().replace("\n", "\n  "))

    inter = tbl2.loc["C(fcategory):C(partner)"]
    print(f"\n  交互作用 F = {inter['F']:.4f}, p = {inter['PR(>F)']:.4f}"
          f"（df {inter['df']:.0f}, {tbl2.loc['Residual', 'df']:.0f}）")
    print(f"  モデル比較でも同じ結論: 残差平方和 {add.ssr:.4f} → {full.ssr:.4f}、"
          f"adj.R² {add.rsquared_adj:.4f} → {full.rsquared_adj:.4f}")
    print(f"  尤度比ではなく F で比べる compare_f_test: "
          f"F = {full.compare_f_test(add)[0]:.4f}, p = {full.compare_f_test(add)[1]:.4f}")

    print("\n  交互作用が有意なので、主効果の行だけを読んではいけない。実際に:")
    piv = df.pivot_table(index="fcategory", columns="partner", values="conformity",
                         aggfunc="mean").reindex(F_LEVELS)[P_LEVELS]
    for lv in F_LEVELS:
        d = piv.loc[lv, "high"] - piv.loc[lv, "low"]
        print(f"    fcategory={lv:<7} 相手が high - low = {d: .4f}")
    print("  ← 相手の地位を上げたときの効き方が、権威主義傾向の水準ごとに違う。"
          "符号まで反転している")
    print(f"  fcategory の主効果は F = {tbl2.loc['C(fcategory)', 'F']:.4f}"
          f"（p = {tbl2.loc['C(fcategory)', 'PR(>F)']:.4f}）と小さいが、")
    print("  これは3水準を partner で平均したら差が消えた、というだけで"
          "「fcategory は効いていない」ではない")

    # --- 交互作用プロット ---
    fig, ax = plots.figure(w=1.1)
    x = np.arange(len(F_LEVELS))
    styles = {"low": (plots.PALETTE["data"], "--", (4, 2.0)),
              "high": (plots.PALETTE["estimate"], "-", ())}
    for p in P_LEVELS:
        color, ls, dashes = styles[p]
        yv = piv[p].to_numpy()
        ax.plot(x, yv, color=color, ls=ls, dashes=dashes or (None, None), lw=1.4,
                marker="o", ms=3.5, zorder=4)
        ax.annotate(f"相手の地位 {p}", xy=(x[-1], yv[-1]), xytext=(4, 0),
                    textcoords="offset points", ha="left", va="center",
                    fontsize=6.2, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(F_LEVELS)
    ax.set_xlim(-0.25, len(F_LEVELS) - 0.35)
    ax.set_xlabel("権威主義傾向 fcategory")
    ax.set_ylabel("同調行動の回数（セル平均）")
    ax.set_title(f"線が平行でない = 交互作用   F = {inter['F']:.3f}, p = {inter['PR(>F)']:.4f}")
    fig.tight_layout()
    plots.save(fig, "fig-14-5-interaction-plot.png")


if __name__ == "__main__":
    main()
