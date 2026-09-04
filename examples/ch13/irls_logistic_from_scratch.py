"""ロジスティック回帰の中身は、重みを付け替えながらの最小二乗。

最尤推定と聞くと最適化ライブラリを呼ぶ話に見えるが、GLM の当てはめは
**反復重み付き最小二乗（IRLS）**という、第12章の正規方程式の使い回しでできている。
毎回、作業応答 $z = \\eta + (y-\\mu)/\\mu'$ と重み $W = \\mu(1-\\mu)$ を作り直して
$b = (X^\\top W X)^{-1} X^\\top W z$ を解く。それだけ。

numpy で 25 行書いて、``statsmodels.Logit`` と係数・標準誤差を突き合わせる。差が
浮動小数の丸めの桁に収まるなら、ライブラリの中身は本当にこの 25 行である。ついでに、
収束が二次で（誤差の桁が毎回おおよそ2倍になって）進むところも見ておく。

    uv run python examples/ch13/irls_logistic_from_scratch.py
"""

import numpy as np
import statsmodels.api as sm

from toukei_tashikame import datasets, glm, plots

N, B_TRUE, SEED = 500, (-1.2, 0.85), 13


def irls_by_hand(X, y, max_iter=25, tol=1e-10):
    """IRLS を手で書く。ロジスティック回帰の当てはめは、これで全部。"""
    b = np.zeros(X.shape[1])
    history = [b.copy()]
    for _ in range(max_iter):
        eta = X @ b                            # 線形予測子
        mu = 1.0 / (1.0 + np.exp(-eta))        # リンクの逆関数（平均）
        w = mu * (1.0 - mu)                    # 重み = 分散関数 V(mu)
        w = np.clip(w, 1e-10, None)            # 0 割りを避ける
        z = eta + (y - mu) / w                 # 作業応答
        xtwx = X.T @ (X * w[:, None])          # X'WX
        b_new = np.linalg.solve(xtwx, X.T @ (w * z))   # 重み付き最小二乗
        history.append(b_new.copy())
        if np.max(np.abs(b_new - b)) < tol:
            b = b_new
            break
        b = b_new
    # 収束点の (X'WX)^{-1} が、そのまま係数の分散共分散になる。
    mu = 1.0 / (1.0 + np.exp(-(X @ b)))
    cov = np.linalg.inv(X.T @ (X * np.clip(mu * (1 - mu), 1e-10, None)[:, None]))
    return b, np.sqrt(np.diag(cov)), np.array(history)


def main() -> None:
    plots.setup()
    X, y, b_true = datasets.logistic_data(N, b=B_TRUE, seed=SEED)

    b_hand, se_hand, hist = irls_by_hand(X, y)
    logit = sm.Logit(y, X).fit(disp=0)
    pkg = glm.irls(X, y, add_const=False, names=["const", "x"])

    print(f"--- 13-4 IRLS を手で書く（n={N}, seed={SEED}, 真値 {np.round(b_true, 2)}）---")
    print("  実装                  切片          傾き        切片のSE     傾きのSE")
    rows = {
        "手書き IRLS": (b_hand, se_hand),
        "statsmodels.Logit": (logit.params, logit.bse),
        "toukei_tashikame.glm": (pkg.b, pkg.se),
    }
    for name, (b, se) in rows.items():
        print(f"  {name:<22}{b[0]:+.8f}  {b[1]:+.8f}  {se[0]:.8f}  {se[1]:.8f}")
    d_b = np.max(np.abs(b_hand - logit.params))
    d_se = np.max(np.abs(se_hand - logit.bse))
    print(f"  手書き vs statsmodels   係数の最大絶対差 {d_b:.3e} / SE の最大絶対差 {d_se:.3e}")
    print(f"  倍精度の刻み eps        {np.finfo(float).eps:.3e}   ← 差は丸めの桁")
    print(f"  真値 {np.round(b_true, 2)} との差は {np.abs(b_hand - b_true).max():.4f}。"
          "これは推定の誤差で、実装の差ではない")

    print(f"\n--- 収束のようす（{len(hist) - 1} 反復で止まった）---")
    print("  反復   切片          傾き          前回からの最大変化")
    for k, b in enumerate(hist):
        step = "—" if k == 0 else f"{np.max(np.abs(hist[k] - hist[k - 1])):.3e}"
        print(f"   {k:>2}   {b[0]:+.10f}  {b[1]:+.10f}   {step}")
    err = np.max(np.abs(hist - b_hand), axis=1)
    print("  ← 1反復ごとに誤差の桁がおよそ倍になる（ニュートン法なので二次収束）")
    print(f"  statsmodels の反復回数 {logit.mle_retvals['iterations']} 回、"
          f"glm.irls は {pkg.n_iter} 回")

    # --- 図: 誤差の対数を反復ごとに ---
    fig, axes = plots.figure(1, 2, w=1.8, h=1.0)
    ax = axes[0]
    k = np.arange(len(err))
    ax.semilogy(k, np.clip(err, 1e-17, None), "o-", ms=3.2,
                color=plots.PALETTE["estimate"], zorder=4)
    ax.axhline(np.finfo(float).eps, color=plots.PALETTE["ink2"], lw=0.8, ls=":",
               zorder=3)
    ax.annotate("倍精度の限界 (eps)", xy=(k[-1], np.finfo(float).eps), ha="right",
                va="bottom", fontsize=6.0, color=plots.PALETTE["ink2"])
    ax.set_xlabel("反復回数")
    ax.set_ylabel("収束点からの最大誤差")
    ax.set_title(f"IRLS は {len(hist) - 1} 反復で丸め誤差まで落ちる")

    ax = axes[1]
    ax.plot(hist[:, 0], hist[:, 1], "o-", ms=3.2, color=plots.PALETTE["estimate"],
            zorder=4)
    for i in (0, 1, 2):
        ax.annotate(f"{i}", xy=(hist[i, 0], hist[i, 1]), xytext=(4, -1),
                    textcoords="offset points", fontsize=6.0,
                    color=plots.PALETTE["estimate"])
    ax.scatter([b_true[0]], [b_true[1]], marker="x", s=30,
               color=plots.PALETTE["truth"], zorder=6)
    ax.annotate(f"真値 ({b_true[0]:g}, {b_true[1]:g})", xy=(b_true[0], b_true[1]),
                xytext=(4, 4), textcoords="offset points", fontsize=6.0,
                color=plots.PALETTE["truth"])
    ax.set_xlabel("切片 $b_0$")
    ax.set_ylabel("傾き $b_1$")
    ax.set_title("係数平面での軌跡（0 から出発）")
    fig.tight_layout()
    plots.save(fig, "fig-13-4-irls-convergence.png")


if __name__ == "__main__":
    main()
