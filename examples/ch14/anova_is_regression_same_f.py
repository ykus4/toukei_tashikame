"""分散分析はダミー変数の重回帰である — 同じデータで F も p も平方和も一致する。

「分散分析」と「回帰分析」は教科書の章が分かれているので別物に見えるが、群を表す
ダミー変数を作って最小二乗を当てれば、出てくる F はまったく同じ値になる。分散分析の
F 検定は、回帰の「全部の係数がまとめて 0 か」という検定そのものである。

同じことの言い換えなので、係数の読み方も対応する。基準群（ここでは ctrl）を除いた
ダミーの係数は、そのまま「基準群との平均差」になる。ANOVA の表には出てこない
「どの群がどれだけ違うか」が、回帰にすると係数として直接読める。

データは R の ``PlantGrowth``（n=30, 3群）。ダウンロードなしで動くよう、30行をここに
直接書いてある。

    uv run python examples/ch14/anova_is_regression_same_f.py
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.anova import anova_lm

# R の datasets::PlantGrowth。乾燥重量 weight と処理群 group（ctrl / trt1 / trt2）。
WEIGHT = [4.17, 5.58, 5.18, 6.11, 4.50, 4.61, 5.17, 4.53, 5.33, 5.14,
          4.81, 4.17, 4.41, 3.59, 5.87, 3.83, 6.03, 4.89, 4.32, 4.69,
          6.31, 5.12, 5.54, 5.50, 5.37, 5.29, 4.92, 6.15, 5.80, 5.26]
GROUP = ["ctrl"] * 10 + ["trt1"] * 10 + ["trt2"] * 10


def main() -> None:
    df = pd.DataFrame({"weight": WEIGHT, "group": GROUP})
    groups = [df.loc[df["group"] == g, "weight"].to_numpy() for g in ["ctrl", "trt1", "trt2"]]

    print(f"--- 14-3 PlantGrowth（n={len(df)}, {df['group'].nunique()}群）---")
    print(df.groupby("group")["weight"].agg(["count", "mean", "std"]).round(4).to_string())

    # --- (A) 一元配置分散分析 ---
    aov_scipy = stats.f_oneway(*groups)

    # --- (B) ダミー変数の重回帰。C(group) が ctrl を基準にしたダミー2本を作る ---
    fit = smf.ols("weight ~ C(group)", data=df).fit()
    aov_table = anova_lm(fit, typ=1)

    print("\n  (A) 一元配置分散分析（scipy.stats.f_oneway）")
    print(f"      F = {aov_scipy.statistic:.4f}   p = {aov_scipy.pvalue:.5f}")
    print("\n  (B) ダミー変数の重回帰（statsmodels OLS）の全体 F")
    print(f"      F = {fit.fvalue:.4f}   p = {fit.f_pvalue:.5f}   "
          f"df = ({fit.df_model:.0f}, {fit.df_resid:.0f})")
    print(f"\n      F の差 {abs(aov_scipy.statistic - fit.fvalue):.2e}   "
          f"p の差 {abs(aov_scipy.pvalue - fit.f_pvalue):.2e}")
    print("      ← 別の関数を呼んでいるだけで、計算しているものは同一")

    print("\n  分散分析表（anova_lm, Type I）— 平方和も一致する")
    print("      " + aov_table.round(4).to_string().replace("\n", "\n      "))
    print(f"      回帰の SSE = {fit.ssr:.4f} / 回帰が説明した SS = {fit.ess:.4f}")

    print("\n  回帰係数は、そのまま群平均差になっている")
    print("      係数              推定値      基準群との実際の差")
    base = groups[0].mean()
    names = {"Intercept": ("ctrl の平均", base),
             "C(group)[T.trt1]": ("trt1 - ctrl", groups[1].mean() - base),
             "C(group)[T.trt2]": ("trt2 - ctrl", groups[2].mean() - base)}
    for key, (label, direct) in names.items():
        print(f"      {key:<18}{fit.params[key]: .4f}   {label} = {direct: .4f}"
              f"   （差 {abs(fit.params[key] - direct):.1e}）")

    print("\n  切片は基準群の平均そのもの。だから「係数が 0 か」の t 検定は"
          "「その群と ctrl の平均が同じか」の検定になる")
    print(f"      trt1: t = {fit.tvalues['C(group)[T.trt1]']:.4f}, "
          f"p = {fit.pvalues['C(group)[T.trt1]']:.4f}")
    print(f"      trt2: t = {fit.tvalues['C(group)[T.trt2]']:.4f}, "
          f"p = {fit.pvalues['C(group)[T.trt2]']:.4f}")
    print("      ただしこの2つは補正なしの比較なので、群が増えれば 14-1 の多重比較の問題に戻る")
    print(f"\n  R² = {fit.rsquared:.4f} は分散分析の η² と同じ量"
          f"（SSB/SST = {aov_table['sum_sq'].iloc[0] / aov_table['sum_sq'].sum():.4f}）")
    print("  分散分析と回帰を「使い分ける」必要はない。同じ当てはめの、"
          "違う要約の出し方だと思えばよい")

    assert np.isclose(aov_scipy.statistic, fit.fvalue)


if __name__ == "__main__":
    main()
