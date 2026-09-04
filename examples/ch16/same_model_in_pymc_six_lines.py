"""@slow — 同じモデルを PyMC で書き直す。モデルの記述は6行。

16-2 で手書きした MH と**同じデータ・同じモデル・同じ事前**を PyMC に渡す。書くのは
「事前はこれ、尤度はこれ、観測はこれ」の3つだけで、提案幅の調整も受容率の見張りも
自分ではやらない。サンプラー（NUTS）が勾配を使って自動で歩幅を決める。

手書き・PyMC・共役解の三者が同じ事後を出すことを確認する。ここで一致していなければ、
以降の章で PyMC が出す数字を信じる理由がない。

    uv sync --extra slow
    uv run python examples/ch16/same_model_in_pymc_six_lines.py
"""

import arviz as az
import numpy as np
import pymc as pm
from scipy import stats

from toukei_tashikame import bayes, plots, sim

MU_TRUE, SIGMA, N = 2.0, 1.0, 50      # 16-2 と同一の設定
SEED = 162
PRIOR_MU, PRIOR_SD = 0.0, 10.0
DRAWS, TUNE, CHAINS = 1_000, 1_000, 4
MH_ITER, MH_BURN, MH_STEP = 10_000, 1_000, 0.50
HDI_MASS = 0.94


def log_posterior(mu: float, y: np.ndarray) -> float:
    """手書き MH に渡す、正規化していない事後の対数（16-2 と同一）。"""
    return float(-0.5 * ((mu - PRIOR_MU) / PRIOR_SD) ** 2
                 - 0.5 * np.sum(((y - mu) / SIGMA) ** 2))


def draw(pymc_draws, mh_draws, exact_mean, exact_sd, density) -> None:
    fig, axes = plots.figure(1, 2, w=2.0, h=1.0, gridspec_kw={"width_ratios": [1.6, 1.0]})
    pal = plots.PALETTE

    ax = axes[0]
    grid = np.linspace(exact_mean - 4.5 * exact_sd, exact_mean + 4.5 * exact_sd, 400)
    ax.plot(grid, stats.norm.pdf(grid, exact_mean, exact_sd), color=pal["truth"],
            lw=1.4, zorder=5)
    ax.hist(pymc_draws, bins=60, density=True, color=pal["posterior"], alpha=0.35,
            lw=0, zorder=2)
    ax.hist(mh_draws, bins=60, density=True, histtype="step", color=pal["data"],
            lw=0.9, zorder=3)
    ax.annotate("共役解（解析）", xy=(exact_mean + 1.6 * exact_sd, 1.6), fontsize=6.0,
                color=pal["truth"])
    ax.annotate(f"PyMC / NUTS {pymc_draws.size:,} 本", xy=(0.03, 0.97),
                xycoords="axes fraction", va="top", fontsize=6.0, color=pal["posterior"])
    ax.annotate(f"手書き MH {mh_draws.size:,} 本（枠線）", xy=(0.03, 0.87),
                xycoords="axes fraction", va="top", fontsize=6.0, color=pal["data"])
    ax.set_xlabel("$\\mu$")
    ax.set_ylabel("事後密度")
    ax.set_title("① 三者が同じ事後に重なる")

    ax = axes[1]
    names = ["手書き MH\n(step 固定)", "PyMC\n(NUTS)"]
    ax.bar(names, density, color=[pal["data"], pal["posterior"]], width=0.55)
    for i, v in enumerate(density):
        ax.annotate(f"{v:.2f}", xy=(i, v), ha="center", va="bottom", fontsize=6.0,
                    color=pal["ink2"])
    ax.set_ylim(0, max(density) * 1.25)
    ax.set_ylabel("ESS / 標本数")
    ax.set_title("② 標本1本の濃さ（自己相関の少なさ）")

    plots.save(fig, "fig-16-4-pymc-vs-handwritten.png")


def main() -> None:
    plots.setup()
    y = np.random.default_rng(SEED).normal(MU_TRUE, SIGMA, size=N)
    prec = 1.0 / PRIOR_SD**2 + N / SIGMA**2
    exact_mean = float((PRIOR_MU / PRIOR_SD**2 + y.sum() / SIGMA**2) / prec)
    exact_sd = float(np.sqrt(1.0 / prec))

    # ---- ここが本体。モデルの記述は6行 ------------------------------------
    with pm.Model():                                                # 1 箱を作る
        mu = pm.Normal("mu", mu=PRIOR_MU, sigma=PRIOR_SD)           # 2 事前
        pm.Normal("y", mu=mu, sigma=SIGMA, observed=y)              # 3 尤度と観測
        with sim.Timer("  PyMC のサンプリング") as t_pymc:            # （計測用）
            idata = pm.sample(DRAWS, tune=TUNE, chains=CHAINS,      # 4 回す
                              random_seed=SEED + 2, progressbar=False)
    summary = az.summary(idata, hdi_prob=HDI_MASS)                  # 5 要約
    post = idata.posterior["mu"].values.ravel()                     # 6 事後標本
    # ----------------------------------------------------------------------

    with sim.Timer("  手書き MH") as t_mh:
        mh = bayes.metropolis_hastings(lambda m: log_posterior(m, y), init=0.0,
                                       n=MH_ITER, step=MH_STEP, seed=SEED + 1)
    mh_draws = mh.burned(MH_BURN)

    pymc_ess = float(az.ess(idata)["mu"].values)
    mh_ess = bayes.ess(mh_draws)

    print(f"\n--- 真値 μ={MU_TRUE}、σ={SIGMA} 既知、n={N}（seed={SEED}）"
          f"／モデルは 16-2 と同一 ---")
    print(f"  PyMC: {CHAINS} 本の鎖 × {DRAWS:,} draws（tune {TUNE:,}）= "
          f"{post.size:,} 本の事後標本\n")

    print("  arviz.summary():")
    print("    " + summary.to_string().replace("\n", "\n    "))

    hdi_pymc = az.hdi(idata, hdi_prob=HDI_MASS)["mu"].values
    hdi_mh = bayes.credible_interval(mh_draws, HDI_MASS, kind="hdi")
    hdi_exact = stats.norm.ppf([(1 - HDI_MASS) / 2, 0.5 + HDI_MASS / 2],
                               exact_mean, exact_sd)

    print("\n                        PyMC          手書き MH        共役解")
    print(f"  事後平均        {post.mean():>12.4f}{mh_draws.mean():>16.4f}"
          f"{exact_mean:>14.4f}")
    print(f"  事後SD          {post.std(ddof=1):>12.4f}{mh_draws.std(ddof=1):>16.4f}"
          f"{exact_sd:>14.4f}")
    print(f"  {HDI_MASS:.0%} 区間        [{hdi_pymc[0]:.3f}, {hdi_pymc[1]:.3f}]"
          f"   [{hdi_mh[0]:.3f}, {hdi_mh[1]:.3f}]   "
          f"[{hdi_exact[0]:.3f}, {hdi_exact[1]:.3f}]")
    print(f"  事後平均の差（PyMC − 共役解）  {post.mean() - exact_mean:+.4f}")
    print(f"  事後平均の差（MH   − 共役解）  {mh_draws.mean() - exact_mean:+.4f}")

    print(f"\n  R̂ = {float(summary['r_hat'].iloc[0]):.3f}"
          f"、ESS(bulk) = {pymc_ess:,.0f} / {post.size:,} 本"
          f"（{pymc_ess / post.size:.2f} 本ぶん）")
    print(f"  手書き MH の ESS = {mh_ess:,.0f} / {mh_draws.size:,} 本"
          f"（{mh_ess / mh_draws.size:.2f} 本ぶん）")
    print("  NUTS は勾配を使って事後の形に沿って歩くので、同じ本数でも中身が濃い。\n")

    print("  書いた行数はモデル3行 + サンプリング1行。提案幅も受容率も出てこない。")
    print("  ただし「回っている」ことと「正しい」ことは別である。次の 16-5 で、")
    print("  R̂ と ESS と発散が何を捕まえるのかを、わざと壊したモデルで見る。")

    print(f"  所要時間は手書き MH {t_mh.elapsed:.2f} 秒に対し PyMC {t_pymc.elapsed:.2f} 秒。")
    print("  パラメータ1個のモデルでは手書きのほうが速い。NUTS の元が取れるのは")
    print("  パラメータが増えて、提案幅の手調整が効かなくなってからである（16-5, 16-7）。")
    draw(post, mh_draws, exact_mean, exact_sd,
         [mh_ess / mh_draws.size, pymc_ess / post.size])


if __name__ == "__main__":
    main()
