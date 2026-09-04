"""0/1 の y に直線を当てると、確率の外側を予測する。

y が 0 か 1 しか取らないとき、$E[y|x]$ はそのまま「1 になる確率」である。だから
線形確率モデル（LPM）は「確率を $b_0 + b_1 x$ で予測する」と言っているのと同じで、
直線は上にも下にも無限に伸びるから、いつかは 0 を下回り 1 を超える。

壊れ方は2つある。**予測が [0,1] を出る**ことと、**残差の分散が x に依存する**こと。
後者は等分散の仮定の破れなので、係数の標準誤差も信用できなくなる。ロジスティック
回帰はこの2つを同時に直す——リンク関数で値域を、二項分布で分散の形を決める。

    uv run python examples/ch13/linear_regression_on_binary_y_breaks.py
"""

from itertools import pairwise

import numpy as np
import statsmodels.api as sm

from toukei_tashikame import datasets, glm, plots

N, B_TRUE, SEED = 400, (-1.0, 0.8), 131


def out_of_range(pred: np.ndarray) -> float:
    """予測が確率として成立しない割合。0 未満か 1 超え。"""
    return float(((pred < 0.0) | (pred > 1.0)).mean())


def main() -> None:
    plots.setup()
    X, y, b_true = datasets.logistic_data(N, b=B_TRUE, seed=SEED)
    x = X[:, 1]

    print(f"--- 13-1 0/1 の y に OLS（n={N}, seed={SEED}, 真の係数 {np.round(b_true, 2)}）---")
    print(f"  y の中身は {np.unique(y)} の2値だけ、1 の割合 {y.mean():.4f}")

    ols = sm.OLS(y, X).fit()
    pred = ols.fittedvalues
    print(f"  OLS の係数            切片 {ols.params[0]:.4f} / 傾き {ols.params[1]:.4f}")
    print(f"  予測値の最小・最大    {pred.min():.4f} / {pred.max():.4f}")
    print(f"  [0,1] の外に出た割合  {out_of_range(pred):.4f}"
          f"（{int(((pred < 0) | (pred > 1)).sum())} / {N} 人）")
    print("  ← 「確率 -0.18」を意味のある予測として使うことはできない")

    # 手元のデータに収まっていても、外挿すれば必ず出る。直線は止まらない。
    grid = np.linspace(-4.0, 4.0, 801)
    line = ols.params[0] + ols.params[1] * grid
    x_neg = (0.0 - ols.params[0]) / ols.params[1]
    x_over = (1.0 - ols.params[0]) / ols.params[1]
    print(f"\n  直線が 0 を割る x      {x_neg:.3f}（観測された x の最小 {x.min():.3f}）")
    print(f"  直線が 1 を超える x    {x_over:.3f}（観測された x の最大 {x.max():.3f}）")
    print(f"  x を [-4, 4] で動かすと外に出る割合 {out_of_range(line):.4f}")

    # 同じデータにロジスティック回帰。リンク関数が値域を (0,1) に閉じ込める。
    fit = glm.irls(X, y, family="binomial", add_const=False, names=["const", "x"])
    p_hat = fit.predict()
    print(f"\n  ロジスティック回帰の係数  切片 {fit.b[0]:.4f} / 傾き {fit.b[1]:.4f}"
          f"（真値 {b_true[0]:.1f} / {b_true[1]:.1f}）")
    print(f"  予測確率の最小・最大      {p_hat.min():.4f} / {p_hat.max():.4f}")
    print(f"  [0,1] の外に出た割合      {out_of_range(p_hat):.4f}   ← 構造的に 0")

    # --- 2つめの壊れ方: 残差の分散が x に依存する ---
    resid = y - pred
    bp_stat, bp_p, _, _ = sm.stats.diagnostic.het_breuschpagan(resid, X)
    print(f"\n--- 残差の分散（Breusch-Pagan 検定 統計量 {bp_stat:.3f}, p={bp_p:.4g}）---")
    print("   x の範囲        残差の分散    p(1-p) の理論値")
    edges = np.quantile(x, np.linspace(0, 1, 6))
    centers, var_emp, var_th = [], [], []
    for lo, hi in pairwise(edges):
        m = (x >= lo) & (x <= hi)
        p_bin = float(glm.expit(X[m] @ fit.b).mean())
        centers.append(float(x[m].mean()))
        var_emp.append(float(resid[m].var(ddof=1)))
        var_th.append(p_bin * (1 - p_bin))
        print(f"  [{lo:+.2f}, {hi:+.2f}]      {var_emp[-1]:.4f}        {var_th[-1]:.4f}")
    print(f"  分散の最大 / 最小 = {max(var_emp) / min(var_emp):.2f} 倍。"
          "等分散を仮定した標準誤差は、この差を見ていない")

    # --- 図 ---
    fig, axes = plots.figure(1, 2, w=1.8, h=1.0)
    ax = axes[0]
    ax.axhspan(1.0, 1.35, color=plots.PALETTE["reject"], alpha=0.18, lw=0, zorder=0)
    ax.axhspan(-0.45, 0.0, color=plots.PALETTE["reject"], alpha=0.18, lw=0, zorder=0)
    ax.scatter(x, y + 0.0, s=6, color=plots.PALETTE["data"], alpha=0.35, lw=0, zorder=2)
    ax.plot(grid, line, color=plots.PALETTE["estimate"], lw=1.3, zorder=4)
    ax.plot(grid, glm.expit(fit.b[0] + fit.b[1] * grid), color=plots.PALETTE["truth"],
            lw=1.2, ls="--", dashes=(4, 2.0), zorder=5)
    ax.annotate("OLS（直線）", xy=(-3.9, ols.params[0] + ols.params[1] * -3.9),
                xytext=(2, -8), textcoords="offset points", fontsize=6.0,
                color=plots.PALETTE["estimate"])
    ax.annotate("ロジスティック", xy=(1.6, glm.expit(fit.b[0] + fit.b[1] * 1.6)),
                xytext=(2, 4), textcoords="offset points", fontsize=6.0,
                color=plots.PALETTE["truth"])
    ax.annotate("確率として成立しない領域", xy=(-3.9, -0.40), fontsize=5.8,
                color=plots.PALETTE["reject"], ha="left", va="bottom")
    ax.set_ylim(-0.45, 1.35)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$（0/1）と予測確率")
    ax.set_title(f"直線は止まらない（外に出た観測 {out_of_range(pred):.1%}）")

    ax = axes[1]
    ax.scatter(centers, var_emp, s=14, color=plots.PALETTE["data"], zorder=3)
    ax.plot(centers, var_emp, color=plots.PALETTE["data"], lw=0.9, zorder=3)
    ax.plot(centers, var_th, color=plots.PALETTE["truth"], lw=1.2, ls="--",
            dashes=(4, 2.0), zorder=4)
    ax.annotate("二項分布の分散 $p(1-p)$", xy=(centers[-2], var_th[-2]),
                xytext=(-6, 6), textcoords="offset points", fontsize=6.0,
                color=plots.PALETTE["truth"], ha="right")
    ax.annotate("OLS 残差の分散", xy=(centers[0], var_emp[0]), xytext=(4, -8),
                textcoords="offset points", fontsize=6.0, color=plots.PALETTE["ink2"])
    ax.set_xlabel("$x$（5分位の中心）")
    ax.set_ylabel("残差の分散")
    ax.set_title(f"分散は x で動く（BP 検定 p={bp_p:.3g}）")
    fig.tight_layout()
    plots.save(fig, "fig-13-1-lpm-out-of-range.png")


if __name__ == "__main__":
    main()
