"""GLM は3部品。分布とリンクを差し替えるだけで、別名の回帰になる。

一般化線形モデルは（1）確率分布、（2）線形予測子 $\\eta = Xb$、（3）リンク関数
$g(\\mu)=\\eta$ の3つでできている。ロジスティック回帰・プロビット回帰・線形確率モデル
は、（1）を二項分布に固定したまま（3）だけを差し替えたもので、モデルの本体は
1つしかない。同じデータに3本当てて、それを目で確かめる。

そのうえでロジットの係数を読む。$e^b$ は**オッズ比**であって確率比ではない。「1.5倍」が
何の1.5倍なのかを分解すると、同じ係数でも基準の確率によって確率の増分がまったく違う
ことが分かる。オッズ比だけを見て「1.5倍増える」と言うのは、たいてい言い過ぎである。

    uv run python examples/ch13/glm_three_parts_swap_link.py
"""

import warnings

import numpy as np
import statsmodels.api as sm

from toukei_tashikame import datasets, glm

N, B_TRUE, SEED = 400, (-1.0, 0.8), 131
LINKS = {
    "logit": sm.families.links.Logit(),
    "probit": sm.families.links.Probit(),
    "identity": sm.families.links.Identity(),
}


def fit_link(X, y, link) -> sm.GLM:
    """分布は二項のまま、リンクだけ差し替えて当てる。"""
    model = sm.GLM(y, X, family=sm.families.Binomial(link=link))
    return model.fit(start_params=np.array([y.mean(), 0.1]))


def main() -> None:
    X, y, b_true = datasets.logistic_data(N, b=B_TRUE, seed=SEED)

    print(f"--- 13-2 分布は二項のまま、リンクだけ差し替える（n={N}, seed={SEED}）---")
    print("  リンク      切片      傾き      対数尤度     AIC      予測が(0,1)外")
    fits = {}
    # identity は二項分布の定義域を守らない。警告が出ること自体が結論なので、
    # 表示だけ抑えて中身は最後まで通す。
    with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
        warnings.simplefilter("ignore")
        for name, link in LINKS.items():
            res = fit_link(X, y, link)
            fits[name] = res
            p = res.fittedvalues
            outside = int(((p < 0) | (p > 1)).sum())
            aic = "  計算不能" if not np.isfinite(res.aic) else f"{res.aic:9.3f}"
            llf = "  計算不能" if not np.isfinite(res.llf) else f"{res.llf:9.3f}"
            print(f"  {name:<10}{res.params[0]:+8.4f}  {res.params[1]:+8.4f}  {llf}  {aic}"
                  f"      {outside:3d} / {N}")

    # identity の対数尤度が nan なのは、確率でないものを確率として対数に入れたから。
    p_id = np.clip(fits["identity"].fittedvalues, 1e-8, 1 - 1e-8)
    llf_clip = float(np.sum(y * np.log(p_id) + (1 - y) * np.log(1 - p_id)))
    print("\n  identity の対数尤度が nan なのは、予測が負になって log が引けないから。")
    print(f"  予測を [1e-8, 1-1e-8] に丸めれば {llf_clip:.3f}"
          f"（AIC {2 * 2 - 2 * llf_clip:.3f}）まで下がる。丸めなければ定義できない")
    print(f"  logit と probit の AIC 差 {abs(fits['logit'].aic - fits['probit'].aic):.3f}"
          "   ← 2つのリンクは、当てはまりでは区別がつかない")

    print("\n--- 係数の目盛りが違うだけ（傾きを比べる）---")
    b_logit, b_probit = fits["logit"].params[1], fits["probit"].params[1]
    print(f"  logit の傾き / probit の傾き = {b_logit / b_probit:.4f}"
          f"   ← 理論値 {np.pi / np.sqrt(3):.4f}（ロジスティック分布と正規分布の SD 比）")
    p_hat = fits["logit"].fittedvalues
    ame = float(np.mean(p_hat * (1 - p_hat)) * b_logit)
    print(f"  logit の平均限界効果 {ame:+.4f} = 平均 p(1-p) × 傾き")
    print(f"  identity の傾き      {fits['identity'].params[1]:+.4f}"
          "   ← 線形確率モデルの係数は「平均的な確率の増分」に近い")

    print("\n--- 13-3 オッズ比 1.5倍は、何の 1.5 倍か ---")
    tab = glm.odds_ratio_table(
        glm.irls(X, y, add_const=False, names=["const", "x"]), conf=0.95)
    print(tab.round(4).to_string())
    or_x = float(tab.loc["x", "OR"])
    lo, hi = float(tab.loc["x", "OR_lo"]), float(tab.loc["x", "OR_hi"])
    print(f"\n  x が 1 増えると、オッズが {or_x:.3f} 倍（95%CI [{lo:.3f}, {hi:.3f}]）")
    print("  「オッズ」= p/(1-p)。確率ではない。同じ 1 倍でも基準の確率で増分が変わる:")
    print("   基準の確率 p0   基準のオッズ   後のオッズ   後の確率 p1   確率の差")
    for p0 in (0.01, 0.05, 0.10, 0.30, 0.50, 0.80):
        odds0 = p0 / (1 - p0)
        odds1 = odds0 * or_x
        p1 = odds1 / (1 + odds1)
        print(f"      {p0:5.2f}         {odds0:7.4f}      {odds1:7.4f}       {p1:.4f}"
              f"      {p1 - p0:+.4f}")
    print("  ← 確率の差は p0=0.50 付近で最大 になり、端では小さい。"
          "オッズ比一定は「確率が一定倍」ではない")
    print(f"  真値との対比: 傾きの真値 {b_true[1]:.2f} → 真のオッズ比 {np.exp(b_true[1]):.3f}"
          f"（推定 {or_x:.3f}、95%CI に真値は{'入る' if lo <= np.exp(b_true[1]) <= hi else '入らない'}）")


if __name__ == "__main__":
    main()
