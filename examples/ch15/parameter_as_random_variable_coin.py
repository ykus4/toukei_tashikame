"""「事前 × 尤度 ∝ 事後」を、式ではなく数え上げで確かめる。

ベイズの式を信じる必要はない。事前から $\\theta$ を引いて、その $\\theta$ でデータを作り、
**手元の観測とぴったり一致した回だけ残す**。残った $\\theta$ の山が事後分布である。
掛け算も正規化定数も出てこない、棄却サンプリングという素朴な手続きだけで済む。

ここで起きているのは、$\\theta$ を「固定された未知の定数」ではなく「引いてくる確率変数」
として扱う、という視点の切り替えである。頻度論との違いは技術ではなく、この一点にある。

    uv run python examples/ch15/parameter_as_random_variable_coin.py
"""

import numpy as np
from scipy import stats

from toukei_tashikame import bayes, plots

N_FLIPS, K_OBS = 100, 30      # 100回投げて30回表、という観測
PRIOR_A, PRIOR_B = 1.0, 1.0   # 事前 Beta(1,1) = 一様
DRAWS, SEED = 200_000, 152


def rejection_sample(rng) -> np.ndarray:
    """事前から引く → データを作る → 観測と一致した θ だけ残す。これだけ。"""
    theta = rng.beta(PRIOR_A, PRIOR_B, size=DRAWS)   # ① 事前から引く
    k = rng.binomial(N_FLIPS, theta)                 # ② その θ でデータを作る
    return theta[k == K_OBS]                         # ③ 観測と一致した回だけ残す


def draw(kept: np.ndarray, post: bayes.BetaPosterior) -> None:
    fig, axes = plots.figure(1, 2, w=2.0, h=0.95)
    pal = plots.PALETTE
    grid = np.linspace(0.0, 1.0, 600)

    ax = axes[0]
    prior = stats.beta.pdf(grid, PRIOR_A, PRIOR_B)
    lik = stats.binom.pmf(K_OBS, N_FLIPS, grid)
    lik = lik / np.trapezoid(lik, grid)              # 形を見るために面積を1に揃える
    plots.prior_posterior(ax, grid, prior=prior, likelihood=lik, posterior=post.pdf(grid))
    ax.annotate("事前 Beta(1,1)", xy=(0.62, 1.4), fontsize=6.0, color=pal["prior"])
    ax.annotate("尤度（正規化済み）", xy=(0.42, 5.0), fontsize=6.0, color=pal["data"])
    ax.annotate(f"事後 Beta({post.a:g}, {post.b:g})", xy=(0.42, 8.2), fontsize=6.0,
                color=pal["posterior"])
    ax.set_xlim(0.0, 0.8)
    ax.set_xlabel("$\\theta$（表の出る確率）")
    ax.set_ylabel("密度")
    ax.set_title("① 事前 × 尤度 ∝ 事後")

    ax = axes[1]
    plots.sim_hist(ax, kept, theory=(grid, post.pdf(grid)), bins=45,
                   theory_label=f"共役解 Beta({post.a:g}, {post.b:g})")
    ax.set_xlim(0.15, 0.5)
    ax.set_xlabel("$\\theta$")
    ax.set_ylabel("密度")
    ax.set_title(f"② 棄却サンプリングで残った {kept.size:,} 本")

    plots.save(fig, "fig-15-2-prior-likelihood-posterior.png")


def main() -> None:
    plots.setup()
    rng = np.random.default_rng(SEED)
    kept = rejection_sample(rng)
    post = bayes.beta_binomial(K_OBS, N_FLIPS, PRIOR_A, PRIOR_B)

    ks = stats.kstest(kept, "beta", args=(post.a, post.b))
    lo, hi = np.quantile(kept, [0.025, 0.975])
    c_lo, c_hi = post.interval(0.95)

    print(f"--- 事前 Beta({PRIOR_A:g}, {PRIOR_B:g}) から {DRAWS:,} 回引き、"
          f"{N_FLIPS} 回中 {K_OBS} 回表と一致した θ だけ残す ---\n")
    print(f"  残った標本          {kept.size:,} 本（採択率 {kept.size / DRAWS:.4f}）")
    print(f"  理論上の採択率      {stats.betabinom.pmf(K_OBS, N_FLIPS, PRIOR_A, PRIOR_B):.4f}"
          "   ← 一様事前なら 1/(n+1) に等しい\n")
    print("                          棄却サンプリング      共役解")
    print(f"  平均                {kept.mean():>16.5f}{post.mean:>13.5f}")
    print(f"  標準偏差            {kept.std(ddof=1):>16.5f}"
          f"{np.sqrt(post.a * post.b / ((post.a + post.b) ** 2 * (post.a + post.b + 1))):>13.5f}")
    print(f"  95%区間の下端       {lo:>16.5f}{c_lo:>13.5f}")
    print(f"  95%区間の上端       {hi:>16.5f}{c_hi:>13.5f}")
    print(f"\n  KS 統計量 {ks.statistic:.4f}（p = {ks.pvalue:.3f}）"
          f" — 残った標本は Beta({post.a:g}, {post.b:g}) と区別がつかない\n")

    print("ここで掛け算は一度もしていない。事前から引いて、データを作って、合わなかった")
    print("ものを捨てただけである。それでベイズの公式と同じ分布が出てくる。")
    print("θ を確率変数として扱う、とはこういうことを言っている。")
    print(f"（採択率が {kept.size / DRAWS:.1%} しかないのがこの方法の限界で、"
          "データが増えるほど")
    print("  一致しなくなる。だから実務では MCMC を使う——第16章の動機はここにある）")
    draw(kept, post)


if __name__ == "__main__":
    main()
