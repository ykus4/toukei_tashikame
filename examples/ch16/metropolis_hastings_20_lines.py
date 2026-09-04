"""Metropolis–Hastings を自分の手で書く。numpy だけ、20行。

MCMC はブラックボックスではない。「今いる場所の近くに1つ提案を出し、事後密度の比で
受け入れるか決める」——それだけである。提案が対称（正規）なら提案密度の比は1に落ちて
式から消え、残るのは事後の比だけになる。正規化定数は比を取ると消えるので、**分母の
積分を一度も計算せずに事後から標本が引ける**。これが 16-1 の行き詰まりの抜け道である。

答え合わせのために、わざと共役解の分かるモデル（分散既知の正規分布の平均）に当てる。
手書きの鎖が解析解に重なることを確認してから、次の節で PyMC に渡す。

    uv run python examples/ch16/metropolis_hastings_20_lines.py
"""

import numpy as np
from scipy import stats

from toukei_tashikame import bayes, plots

MU_TRUE, SIGMA, N = 2.0, 1.0, 50      # 真値 2.0、分散は既知とする
SEED = 162
PRIOR_MU, PRIOR_SD = 0.0, 10.0        # 弱情報事前 N(0, 10^2)
N_ITER, BURN, STEP = 10_000, 1_000, 0.50


def log_posterior(mu: float, y: np.ndarray) -> float:
    """正規化していない事後の対数。事前の対数 + 尤度の対数。分母は要らない。"""
    log_prior = -0.5 * ((mu - PRIOR_MU) / PRIOR_SD) ** 2
    log_lik = -0.5 * np.sum(((y - mu) / SIGMA) ** 2)
    return float(log_prior + log_lik)


def metropolis_hastings(logpost, init, n, step, seed):
    """これが全部。ループの中は4行しかない。"""
    rng = np.random.default_rng(seed)
    x = float(init)
    log_p = logpost(x)
    chain = np.empty(n)
    accepted = 0
    for i in range(n):
        proposal = x + rng.normal(0.0, step)          # ① 近くに1つ提案する
        log_q = logpost(proposal)                     # ② 提案先の事後（比例分）を測る
        if np.log(rng.random()) < log_q - log_p:      # ③ 比で受け入れを決める
            x, log_p = proposal, log_q                #    受け入れたら動く
            accepted += 1
        chain[i] = x                                  # ④ 動かなくても今の場所を記録する
    return chain, accepted / n


def analytic_posterior(y: np.ndarray) -> tuple[float, float]:
    """共役解。正規（既知分散）の平均に正規事前を当てると、事後も正規になる。"""
    prec = 1.0 / PRIOR_SD**2 + y.size / SIGMA**2
    mean = (PRIOR_MU / PRIOR_SD**2 + y.sum() / SIGMA**2) / prec
    return float(mean), float(np.sqrt(1.0 / prec))


def draw(chain, exact_mean, exact_sd) -> None:
    fig, axes = plots.figure(1, 2, w=2.0, h=1.0, gridspec_kw={"width_ratios": [1.6, 1.0]})
    pal = plots.PALETTE

    ax = axes[0]
    ax.plot(np.arange(BURN), chain[:BURN], color=pal["data"], lw=0.4, zorder=3)
    ax.plot(np.arange(BURN, N_ITER), chain[BURN:], color=pal["posterior"], lw=0.4,
            zorder=3)
    ax.axvspan(0, BURN, color=pal["reject"], alpha=0.18, lw=0, zorder=1)
    ax.annotate(f"バーンイン {BURN} 回\n（初期値 {chain[0]:.1f} の影響が抜けるまで）",
                xy=(BURN * 1.3, chain.min()), fontsize=6.0, color=pal["reject"],
                ha="left", va="bottom")
    plots.mark_truth(ax, exact_mean, f"共役解の事後平均 = {exact_mean:.4f}", axis="y")
    ax.set_xlabel("反復")
    ax.set_ylabel("$\\mu$")
    ax.set_title("① トレース — 毛虫のように見えれば動けている")

    ax = axes[1]
    kept = chain[BURN:]
    grid = np.linspace(exact_mean - 4.5 * exact_sd, exact_mean + 4.5 * exact_sd, 400)
    plots.sim_hist(ax, kept, theory=(grid, stats.norm.pdf(grid, exact_mean, exact_sd)),
                   bins=45, theory_label="共役解")
    ax.annotate(f"MH {kept.size:,} 本\n平均 {kept.mean():.4f}\nSD {kept.std(ddof=1):.4f}",
                xy=(0.03, 0.97), xycoords="axes fraction", va="top", fontsize=6.0,
                color=pal["posterior"])
    ax.set_xlabel("$\\mu$")
    ax.set_ylabel("事後密度")
    ax.set_title("② 手書きの鎖が解析解に重なる")

    plots.save(fig, "fig-16-2-mh-trace-and-posterior.png")


def main() -> None:
    plots.setup()
    y = np.random.default_rng(SEED).normal(MU_TRUE, SIGMA, size=N)
    exact_mean, exact_sd = analytic_posterior(y)

    chain, rate = metropolis_hastings(lambda m: log_posterior(m, y), init=0.0,
                                      n=N_ITER, step=STEP, seed=SEED + 1)
    kept = chain[BURN:]
    lo, hi = bayes.credible_interval(kept, 0.94, kind="hdi")
    exact_lo, exact_hi = stats.norm.ppf([0.03, 0.97], exact_mean, exact_sd)

    print(f"--- 真値 μ={MU_TRUE}、σ={SIGMA} 既知、n={N}（seed={SEED}）---")
    print(f"  標本平均 {y.mean():.4f}（真値からのずれ {y.mean() - MU_TRUE:+.4f}）")
    print(f"  MH: {N_ITER:,} 反復、提案幅 step={STEP}、バーンイン {BURN} を捨てる\n")
    print(f"  受容率 {rate:.4f}"
          "   ← 0.2〜0.5 が目安。高すぎるのは動けていない、低すぎるのは跳びすぎ\n")

    print("                        手書き MH        共役解          差")
    print(f"  事後平均        {kept.mean():>16.4f}{exact_mean:>14.4f}"
          f"{abs(kept.mean() - exact_mean):>11.4f}")
    print(f"  事後SD          {kept.std(ddof=1):>16.4f}{exact_sd:>14.4f}"
          f"{abs(kept.std(ddof=1) - exact_sd):>11.4f}")
    print(f"  94% 区間 下端   {lo:>16.4f}{exact_lo:>14.4f}{abs(lo - exact_lo):>11.4f}")
    print(f"  94% 区間 上端   {hi:>16.4f}{exact_hi:>14.4f}{abs(hi - exact_hi):>11.4f}")

    print(f"\n  有効標本数 ESS = {bayes.ess(kept):,.0f}（{kept.size:,} 本の鎖から）")
    print("  隣どうしが相関しているので、10,000 本引いても独立な標本ぶんの情報はない。")
    print("  ここは 16-3 で提案幅を振って見る。\n")
    print("  分母（正規化定数）は一度も計算していない。log_q - log_p の差を取った時点で")
    print("  約分されて消えている。16-1 で詰まった積分を、比に置き換えて避けたのがこの20行。")

    draw(chain, exact_mean, exact_sd)


if __name__ == "__main__":
    main()
