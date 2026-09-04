"""回数を数えるならポアソン回帰。「何あたりの回数か」はオフセットで決まる。

回数データの回帰は、対数リンクのおかげで係数が**率比**になる。$e^b$ が 0.62 なら
「その変数が1増えると回数が 0.62 倍」で、差ではなく倍で読む。

もう1つの部品がオフセットである。観測の機会（露出）が人によって違うとき——ここでは
結婚年数が 0.5 年の人と 23 年の人が同じ列に並んでいる——回数をそのまま比べても
意味がない。$\\log(\\text{結婚年数})$ を係数 1 固定で線形予測子に足すと、モデルは
「回数」ではなく「**結婚1年あたりの回数**」を説明する式に変わる。

statsmodels 同梱の fair（Affairs, n=6366）で、入れる／入れないの2本を当てて比べる。

    uv run python examples/ch13/poisson_regression_offset_per_1000.py
"""

import numpy as np
import statsmodels.api as sm
from scipy import stats

from toukei_tashikame import glm, plots

COLS = ["rate_marriage", "age", "children", "religious"]


def main() -> None:
    plots.setup()
    d = sm.datasets.fair.load_pandas().data
    # affairs は「浮気に費やした時間」の指標なので小数を含む。回数として扱うために丸める。
    y = np.round(d["affairs"]).to_numpy(dtype=float)
    Xdf = sm.add_constant(d[COLS])
    X = Xdf.to_numpy(dtype=float)
    exposure = d["yrs_married"].to_numpy(dtype=float)

    print(f"--- 13-6 ポアソン回帰とオフセット（fair, n={len(d):,}）---")
    print(f"  y（丸めた affairs）平均 {y.mean():.4f} / 分散 {y.var(ddof=1):.4f}"
          f"   ← 平均=分散 のはずが {y.var(ddof=1) / y.mean():.2f} 倍（13-7 へ続く）")
    print(f"  0 の割合 {np.mean(y == 0):.4f}、最大 {y.max():.0f}")
    print(f"  結婚年数 yrs_married  中央値 {np.median(exposure):.1f} 年"
          f"（{exposure.min():.1f} 〜 {exposure.max():.1f}）"
          "   ← 観測の機会がこれだけ違う")

    plain = sm.GLM(y, Xdf, family=sm.families.Poisson()).fit()
    off = sm.GLM(y, Xdf, family=sm.families.Poisson(), offset=np.log(exposure)).fit()

    print("\n  率比 exp(b)（1 なら効果なし）")
    print("  変数              オフセットなし   オフセットあり")
    for i, name in enumerate(["const", *COLS]):
        print(f"  {name:<16}{np.exp(plain.params.iloc[i]):12.4f}   {np.exp(off.params.iloc[i]):12.4f}")
    i_rm = 1 + COLS.index("rate_marriage")
    ci = np.exp(off.conf_int().to_numpy()[i_rm])
    print(f"\n  rate_marriage の率比  {np.exp(plain.params.iloc[i_rm]):.4f}"
          f" → {np.exp(off.params.iloc[i_rm]):.4f}"
          f"（95%CI [{ci[0]:.4f}, {ci[1]:.4f}]）")
    print("  読み方: 結婚満足度が1段階高い人は、結婚1年あたりの回数が"
          f" {np.exp(off.params.iloc[i_rm]):.3f} 倍。"
          f"1 段階で {100 * (1 - np.exp(off.params.iloc[i_rm])):.1f}% 減る")

    print("\n  summary() の抜粋（オフセットあり）")
    tbl = off.summary().tables[1].as_text().splitlines()
    for line in tbl[: 2 + len(COLS) + 2]:
        print("   " + line)

    # --- オフセットは「係数を 1 に固定した共変量」である ---
    Xfree = Xdf.assign(log_yrs=np.log(exposure))
    free = sm.GLM(y, Xfree, family=sm.families.Poisson()).fit()
    b_log, se_log = free.params.iloc[-1], free.bse.iloc[-1]
    zval = (b_log - 1.0) / se_log
    print("\n--- オフセットの仮定を検査する（log(結婚年数) の係数を自由に推定）---")
    print(f"  自由に推定した係数 {b_log:.4f}（SE {se_log:.4f}）")
    print(f"  1 との差の z 値 {zval:.3f}、p = {2 * stats.norm.sf(abs(zval)):.3e}")
    print("  ← 1 から離れているなら、回数は結婚年数に比例していない。"
          "オフセットは便利だが、無検査で入れる仮定ではない")
    print(f"  AIC  オフセットなし {plain.aic:.1f} / あり {off.aic:.1f} / 自由 {free.aic:.1f}")

    # --- 過分散の確認（13-7 への橋渡し）---
    res_off = glm.irls(X, y, family="poisson", add_const=False,
                       offset=np.log(exposure), names=["const", *COLS])
    print(f"\n  Pearson χ²/df  オフセットあり {glm.dispersion(res_off):.3f}"
          f"（statsmodels {off.pearson_chi2 / off.df_resid:.3f}）")
    print("  ← 1 を大きく超えている。標準誤差はこのぶん小さく出すぎている（13-7）")

    # --- 図 ---
    fig, axes = plots.figure(1, 2, w=1.8, h=1.0)
    ax = axes[0]
    levels = np.unique(d["rate_marriage"].to_numpy())
    obs = [y[d["rate_marriage"].to_numpy() == v].mean() for v in levels]
    pr_plain = [plain.fittedvalues[d["rate_marriage"].to_numpy() == v].mean() for v in levels]
    pr_off = [off.fittedvalues[d["rate_marriage"].to_numpy() == v].mean() for v in levels]
    ax.scatter(levels, obs, s=18, color=plots.PALETTE["data"], zorder=5)
    ax.plot(levels, pr_plain, color=plots.PALETTE["estimate"], lw=1.2, zorder=4)
    ax.plot(levels, pr_off, color=plots.PALETTE["reject"], lw=1.2, ls="--",
            dashes=(4, 2.0), zorder=4)
    ax.annotate("実測平均", xy=(levels[0], obs[0]), xytext=(4, 4),
                textcoords="offset points", fontsize=6.0, color=plots.PALETTE["ink2"])
    ax.annotate("オフセットなし", xy=(levels[1], pr_plain[1]), xytext=(4, -9),
                textcoords="offset points", fontsize=6.0, color=plots.PALETTE["estimate"])
    ax.annotate("オフセットあり", xy=(levels[2], pr_off[2]), xytext=(4, 4),
                textcoords="offset points", fontsize=6.0, color=plots.PALETTE["reject"])
    ax.set_xlabel("結婚満足度 rate_marriage")
    ax.set_ylabel("回数の平均")
    ax.set_title("満足度が上がると回数は減る")

    ax = axes[1]
    bins = np.array([0.5, 2.5, 6.0, 10.5, 16.5, 23.0])
    idx = np.digitize(exposure, bins[1:-1], right=True)
    centers, rate_obs, rate_plain, rate_off = [], [], [], []
    for k in range(len(bins) - 1):
        m = idx == k
        centers.append(exposure[m].mean())
        rate_obs.append((y[m] / exposure[m]).mean())
        rate_plain.append((plain.fittedvalues[m] / exposure[m]).mean())
        rate_off.append((off.fittedvalues[m] / exposure[m]).mean())
    ax.scatter(centers, rate_obs, s=18, color=plots.PALETTE["data"], zorder=5)
    ax.plot(centers, rate_plain, color=plots.PALETTE["estimate"], lw=1.2, zorder=4)
    ax.plot(centers, rate_off, color=plots.PALETTE["reject"], lw=1.2, ls="--",
            dashes=(4, 2.0), zorder=4)
    ax.annotate("実測（回数 / 結婚年数）", xy=(centers[1], rate_obs[1]), xytext=(4, 4),
                textcoords="offset points", fontsize=6.0, color=plots.PALETTE["ink2"])
    ax.set_xlabel("結婚年数（5分割の平均）")
    ax.set_ylabel("結婚1年あたりの回数")
    ax.set_title("オフセットは「1年あたり」に目盛りを直す")
    fig.tight_layout()
    plots.save(fig, "fig-13-6-poisson-offset-fit.png")


if __name__ == "__main__":
    main()
