"""最尤推定を「山登り」として書く — 手書きの対数尤度・グリッド探索・最適化・閉じた式。

正規・ベルヌーイ・ポアソンの対数尤度を numpy で直に書き、(1) グリッドで一番高い点を探す
(2) ``scipy.optimize.minimize`` に登らせる (3) 紙で解いた閉じた式、の3つが同じ答えに
なることを見る。最尤推定は魔法ではなく、データが決めた関数の頂上を探しているだけである。

    uv run python examples/ch06/mle_by_hand_and_scipy_optimize.py
"""

import numpy as np
from scipy import optimize, special

from toukei_tashikame import plots

SEED = 26
MU_TRUE, SIGMA_TRUE, N_NORMAL = 50.0, 10.0, 200
P_TRUE, N_BERNOULLI = 0.03, 1_000
LAMBDA_TRUE, N_POISSON = 3.0, 500


def loglik_normal(x, mu, sigma):
    """正規分布の対数尤度。log の中身を手で書き下しただけのもの。"""
    return float(-0.5 * x.size * np.log(2 * np.pi * sigma**2)
                 - np.sum((x - mu) ** 2) / (2 * sigma**2))


def loglik_bernoulli(x, p):
    """ベルヌーイの対数尤度。成功 k 回、失敗 n-k 回。定義域の外は -inf。"""
    if not 0.0 < p < 1.0:
        return -np.inf
    k = float(x.sum())
    return float(k * np.log(p) + (x.size - k) * np.log1p(-p))


def loglik_poisson(x, lam):
    """ポアソンの対数尤度。log(x!) は gammaln で置き換える。定義域の外は -inf。"""
    if lam <= 0.0:
        return -np.inf
    return float(np.sum(x * np.log(lam) - lam - special.gammaln(x + 1)))


def climb(loglik, grid, x0):
    """グリッド探索と minimize の2通りで頂上を探し、``(grid_hat, opt_hat)`` を返す。"""
    curve = np.array([loglik(g) for g in grid])
    grid_hat = float(grid[int(np.argmax(curve))])
    out = optimize.minimize(lambda t: -loglik(float(t[0])), x0=[x0], method="Nelder-Mead",
                            options={"xatol": 1e-12, "fatol": 1e-12})
    return grid_hat, float(out.x[0]), curve


def report(name, grid_hat, opt_hat, closed_hat, truth, step):
    print(f"--- {name} ---")
    print(f"  グリッド探索   {grid_hat:.6f}   （刻み {step:.1e} までしか当たらない）")
    print(f"  minimize      {opt_hat:.6f}")
    print(f"  閉じた式       {closed_hat:.6f}   （真値 {truth:g}）")
    print(f"  最大の差       グリッド vs 閉じた式 {abs(grid_hat - closed_hat):.2e} / "
          f"minimize vs 閉じた式 {abs(opt_hat - closed_hat):.2e}")


def main() -> None:
    plots.setup()
    rng = np.random.default_rng(SEED)
    x_norm = rng.normal(MU_TRUE, SIGMA_TRUE, size=N_NORMAL)
    x_bern = rng.binomial(1, P_TRUE, size=N_BERNOULLI).astype(float)
    x_pois = rng.poisson(LAMBDA_TRUE, size=N_POISSON).astype(float)

    # 正規: σ は最尤値に固定して μ の断面を見る（閉じた式では μ の解が σ に依らない）
    sigma_hat = float(x_norm.std(ddof=0))
    mu_grid = np.linspace(45.0, 55.0, 2001)
    mu_g, mu_o, curve_n = climb(lambda m: loglik_normal(x_norm, m, sigma_hat), mu_grid, 40.0)
    report(f"正規分布の μ（n={N_NORMAL}）", mu_g, mu_o, float(x_norm.mean()), MU_TRUE,
           mu_grid[1] - mu_grid[0])
    print(f"  ついでに σ²: 最尤 {sigma_hat**2:.4f}（ddof=0）/ 不偏 {x_norm.var(ddof=1):.4f}"
          f"（真値 {SIGMA_TRUE**2:g}）"
          f"\n  最尤の σ̂² は不偏分散のちょうど (n-1)/n = {(N_NORMAL - 1) / N_NORMAL:.4f} 倍。"
          "最尤推定量は不偏とは限らない\n")

    p_grid = np.linspace(1e-4, 0.10, 2001)
    p_g, p_o, curve_b = climb(lambda p: loglik_bernoulli(x_bern, p), p_grid, 0.20)
    report(f"ベルヌーイの p（n={N_BERNOULLI}）", p_g, p_o, float(x_bern.mean()), P_TRUE,
           p_grid[1] - p_grid[0])
    print(f"  観測は {int(x_bern.sum())} 回の成功 / {N_BERNOULLI} 回\n")

    lam_grid = np.linspace(1.0, 5.0, 2001)
    lam_g, lam_o, curve_p = climb(lambda lam: loglik_poisson(x_pois, lam), lam_grid, 1.0)
    report(f"ポアソンの λ（n={N_POISSON}）", lam_g, lam_o, float(x_pois.mean()), LAMBDA_TRUE,
           lam_grid[1] - lam_grid[0])
    print("  3つとも「標本平均」に落ちる。分布が違っても頂上の探し方は同じ\n")

    print("--- 読み方 ---")
    print("  グリッドは刻み幅の分だけ必ずずれる。minimize は閉じた式と 1e-6 の桁まで一致する")
    print("  閉じた式があるならそれを使う。無い分布（第13章の IRLS など）でも登り方は同じ")

    fig, axes = plots.figure(1, 3, w=3.0, h=0.95)
    panels = [
        ("正規: μ", mu_grid, curve_n, mu_o, MU_TRUE, "μ"),
        ("ベルヌーイ: p", p_grid, curve_b, p_o, P_TRUE, "p"),
        ("ポアソン: λ", lam_grid, curve_p, lam_o, LAMBDA_TRUE, "λ"),
    ]
    for ax, (title, grid, curve, hat, truth, sym) in zip(axes, panels, strict=True):
        ax.plot(grid, curve, color=plots.PALETTE["data"], lw=1.1)
        ax.axvline(hat, color=plots.PALETTE["estimate"], lw=1.2, ls="--", dashes=(4, 2.0))
        ax.annotate(f"最尤 {sym}̂ = {hat:.4g}", xy=(hat, 0.06), xycoords=("data", "axes fraction"),
                    ha="left", va="bottom", fontsize=6.0, color=plots.PALETTE["estimate"],
                    xytext=(3, 0), textcoords="offset points")
        plots.mark_truth(ax, truth, f"真値 {truth:g}")
        ax.set_ylim(curve.max() - 6 * (curve.max() - np.quantile(curve, 0.5)) / 5, curve.max() + 1)
        ax.set_xlabel(sym)
        ax.set_title(title)
    axes[0].set_ylabel("対数尤度")
    plots.save(fig, "fig-6-4-loglik-curves.png")


if __name__ == "__main__":
    main()
