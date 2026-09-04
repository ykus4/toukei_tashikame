"""事後分布は「刻んで、掛けて、正規化する」だけで作れる。

共役も MCMC も要らない。パラメータの取りうる範囲を格子に刻み、各点で事前と尤度を掛け、
最後に合計で割る。3行である。しかも共役解が存在するモデルなら、その3行が解析解に
小数点以下4桁まで一致することを確かめられる。

グリッド近似は次元が増えると使えなくなる（第16章 16-1 でそこを測る）が、1次元では
これが一番わかりやすい。ベイズを「難しい積分」だと思わないために、ここから入る。

    uv run python examples/ch15/grid_approximation_posterior.py
"""

import numpy as np
from scipy import stats

from toukei_tashikame import bayes, plots

N_TRIALS, K_SUCCESS = 30, 9    # 30回中9成功
PRIOR_A, PRIOR_B = 1.0, 1.0    # 一様事前
N_GRID = 1000
HDI_MASS = 0.94                # arviz の既定に揃える


def grid_hdi(grid: np.ndarray, dens: np.ndarray, mass: float) -> tuple[float, float]:
    """密度の高い点から順に拾って、質量が mass を超えたところで切る。"""
    w = dens / dens.sum()
    order = np.argsort(w)[::-1]
    keep = np.zeros_like(w, dtype=bool)
    keep[order[np.cumsum(w[order]) <= mass]] = True
    keep[order[0]] = True
    inside = grid[keep]
    return float(inside.min()), float(inside.max())


def draw(grid, post_grid, post, hdi) -> None:
    fig, axes = plots.figure(1, 2, w=2.0, h=0.95)
    pal = plots.PALETTE

    ax = axes[0]
    exact = post.pdf(grid)
    ax.fill_between(grid, post_grid, where=(grid >= hdi[0]) & (grid <= hdi[1]),
                    color=pal["posterior"], alpha=0.22, lw=0, zorder=1)
    ax.plot(grid[::20], post_grid[::20], ls="none", marker="o", ms=2.2,
            color=pal["posterior"], zorder=4)
    ax.plot(grid, exact, color=pal["truth"], lw=1.1, ls="--", dashes=(4, 2.0), zorder=5)
    ax.annotate(f"共役解 Beta({post.a:g}, {post.b:g})", xy=(0.52, 3.0), fontsize=6.0,
                color=pal["truth"])
    ax.annotate(f"グリッド {N_GRID} 点", xy=(0.05, 3.6), fontsize=6.0,
                color=pal["posterior"])
    ax.annotate(f"{HDI_MASS:.0%} HDI\n[{hdi[0]:.3f}, {hdi[1]:.3f}]",
                xy=(np.mean(hdi), 0.6), ha="center", fontsize=6.0, color=pal["estimate"])
    ax.set_xlim(0.0, 0.8)
    ax.set_xlabel("$\\theta$")
    ax.set_ylabel("事後密度")
    ax.set_title("① グリッドの点が共役解の曲線に乗る")

    ax = axes[1]
    sizes = np.array([5, 10, 20, 50, 100, 200, 500, 1000, 2000])
    errs = []
    for m in sizes:
        g = np.linspace(0.0, 1.0, m)
        p = bayes.grid_posterior(stats.binom.logpmf(K_SUCCESS, N_TRIALS, g),
                                 stats.beta.pdf(g, PRIOR_A, PRIOR_B), g)
        errs.append(abs(float(np.trapezoid(g * p, g)) - post.mean))
    ax.plot(sizes, errs, color=pal["estimate"], lw=1.2, marker="o", ms=2.6, zorder=4)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("格子点数")
    ax.set_ylabel("事後平均の絶対誤差")
    ax.set_title("② 刻みを細かくすれば誤差は消える（1次元なら安い）")

    plots.save(fig, "fig-15-4-grid-approximation.png")


def main() -> None:
    plots.setup()
    grid = np.linspace(0.0, 1.0, N_GRID)

    # ここが全部。刻んで、対数尤度と事前を足して、正規化する。
    loglik = stats.binom.logpmf(K_SUCCESS, N_TRIALS, grid)
    prior = stats.beta.pdf(grid, PRIOR_A, PRIOR_B)
    post_grid = bayes.grid_posterior(loglik, prior, grid)

    post = bayes.beta_binomial(K_SUCCESS, N_TRIALS, PRIOR_A, PRIOR_B)
    mean_grid = float(np.trapezoid(grid * post_grid, grid))
    sd_grid = float(np.sqrt(np.trapezoid((grid - mean_grid) ** 2 * post_grid, grid)))
    hdi = grid_hdi(grid, post_grid, HDI_MASS)

    samples = post.rvs(200_000, seed=154)
    hdi_exact = bayes.credible_interval(samples, HDI_MASS, kind="hdi")
    sd_exact = np.sqrt(post.a * post.b / ((post.a + post.b) ** 2 * (post.a + post.b + 1)))

    print(f"--- {N_TRIALS} 回中 {K_SUCCESS} 成功、一様事前、格子 {N_GRID} 点 ---\n")
    print("  手書きの中身は3行:")
    print("    loglik = stats.binom.logpmf(k, n, grid)")
    print("    post   = np.exp(loglik) * prior")
    print("    post  /= np.trapezoid(post, grid)\n")
    print("                        グリッド近似        共役解         差")
    print(f"  事後平均        {mean_grid:>16.5f}{post.mean:>14.5f}"
          f"{abs(mean_grid - post.mean):>11.1e}")
    print(f"  事後SD          {sd_grid:>16.5f}{sd_exact:>14.5f}"
          f"{abs(sd_grid - sd_exact):>11.1e}")
    print(f"  {HDI_MASS:.0%} HDI 下端    {hdi[0]:>16.5f}{hdi_exact[0]:>14.5f}"
          f"{abs(hdi[0] - hdi_exact[0]):>11.1e}")
    print(f"  {HDI_MASS:.0%} HDI 上端    {hdi[1]:>16.5f}{hdi_exact[1]:>14.5f}"
          f"{abs(hdi[1] - hdi_exact[1]):>11.1e}")

    maxdiff = float(np.abs(post_grid - post.pdf(grid)).max())
    print(f"\n  密度そのものの最大絶対差  {maxdiff:.1e}")
    print(f"  最尤推定 {K_SUCCESS / N_TRIALS:.4f} に対し、事後平均は "
          f"{post.mean:.5f} と少し 0.5 側に引かれている。")
    print("  一様事前が「1回成功・1回失敗をすでに見た」ぶんだけ効いている\n")

    print("グリッド近似が効くのは、正規化定数を「全部の点を足す」で片づけられるから。")
    print("パラメータが2つなら格子は2乗、5つなら5乗になる。1000点刻みの5次元は")
    print("10^15 点で、もう数えられない。MCMC はここから先の話である。")
    draw(grid, post_grid, post, hdi)


if __name__ == "__main__":
    main()
