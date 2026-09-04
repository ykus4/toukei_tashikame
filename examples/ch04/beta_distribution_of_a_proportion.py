"""割合そのものの分布はベータ分布。10,000回の A/Bテストと、1回ぶんのベータ事後を重ねる。

二項分布は「n 人のうち何人がCVしたか」という**件数**の分布だった（4-2）。知りたいのは
たいていそちらではなく、**割合** $p$ のほうである。件数を n で割った $\\hat{p}$ が
どうばらつくか、そして「1回のテストを見た後で $p$ について何が言えるか」——この2つに
同じ形の答えを与えるのがベータ分布である。

前半は頻度の側。真のCVR 0.03、n=1,000 の A/Bテストを10,000回まわし、$\\hat{p}$ の
散らばりを数え上げる。後半はベイズの側。そのうち1回だけを見て、一様事前 $\\mathrm{Beta}(1,1)$
から $\\mathrm{Beta}(1+k, 1+n-k)$ に更新する。30人がCVしたなら $\\mathrm{Beta}(31, 971)$ である。

2つはほぼ重なる。ほぼ、であって完全ではない。一様事前は「成功1回・失敗1回を先に見た」
のと同じ効き方をするので、CVR 3% のような小さい割合では平均が 1/1002 だけ上に寄る。
その 0.001 が、$\\hat{p}$ の SD 0.0054 に対して無視できない大きさで残る——事前分布が
効いている、とはこういうことである。

    uv run python examples/ch04/beta_distribution_of_a_proportion.py
"""

import numpy as np
from scipy import stats

from toukei_tashikame import datasets, plots, sim

N_USER = 1_000
P_TRUE = 0.03
TRIALS = 10_000
SEED = 17

OBSERVED_CV = 30           # 「ある1回のテスト」で観測されたCV数（= n * p_true）
A_POST, B_POST = 1 + OBSERVED_CV, 1 + (N_USER - OBSERVED_CV)   # 一様事前からの更新


def one_trial(rng) -> int:
    """A/Bテストを1回まわし、A群のCV数を返す。"""
    d = datasets.ab_test(
        n_a=N_USER, n_b=N_USER, p_a=P_TRUE, lift=0.0,
        seed=int(rng.integers(2**31)),
    )
    return int(d.a.sum())


def cdf_gap(phat: np.ndarray, dist) -> float:
    """離散な $\\hat{p}$ の経験CDFと、連続なベータのCDFの最大差。

    $\\hat{p}$ は 1/n 刻みの階段なので、そのまま比べると刻み幅ぶんの差が必ず残る。
    段の中央 $(k+0.5)/n$ で比べる（連続性補正）ことで、刻みのせいの差を落とし、
    分布の位置と幅の違いだけを見る。
    """
    ks = np.unique(np.rint(phat * N_USER).astype(int))
    ecdf = np.array([(phat <= k / N_USER + 1e-12).mean() for k in ks])
    return float(np.abs(ecdf - dist.cdf((ks + 0.5) / N_USER)).max())


def frequentist_side(phat: np.ndarray) -> None:
    print(f"--- 真のCVR {P_TRUE} のテスト（各 {N_USER:,} 人）を {TRIALS:,} 回 ---")
    se = np.sqrt(P_TRUE * (1 - P_TRUE) / N_USER)
    print(f"  p̂ の平均 {phat.mean():.5f}   （真値 {P_TRUE:.5f}）")
    print(f"  p̂ の SD  {phat.std(ddof=1):.5f}   （理論 √(p(1−p)/n) = {se:.5f}）")
    lo, hi = np.quantile(phat, [0.025, 0.975])
    print(f"  p̂ の中央95% [{lo:.4f}, {hi:.4f}]   幅 {hi - lo:.4f}"
          f"（真値の {(hi - lo) / P_TRUE:.0%} ぶん）")
    print(f"  最小 {phat.min():.4f} / 最大 {phat.max():.4f}。"
          "真のCVRは1つなのに、観測されるCVRはこれだけ動く")


def bayesian_side() -> tuple[object, object]:
    post = stats.beta(A_POST, B_POST)
    print(f"\n--- そのうち1回だけを見る。{OBSERVED_CV} 人がCVした ---")
    print(f"  事前 Beta(1, 1)（一様） → 事後 Beta({A_POST}, {B_POST})")
    print(f"  事後の平均 {post.mean():.5f}   = (1+{OBSERVED_CV}) / (2+{N_USER})")
    print(f"  事後の SD  {post.std():.5f}")
    lo, hi = post.ppf(0.025), post.ppf(0.975)
    print(f"  95%信用区間 [{lo:.4f}, {hi:.4f}]")
    print(f"  Pr[p < 0.03] = {post.cdf(0.03):.4f} / Pr[p > 0.035] = {post.sf(0.035):.4f}"
          "   ← 割合そのものについての確率を、そのまま計算できる")

    # 事前を 1 件ぶんも足さない極限。件数がそのまま形になる。
    flat = stats.beta(OBSERVED_CV, N_USER - OBSERVED_CV)
    print(f"\n  比較: 事前を足さない Beta({OBSERVED_CV}, {N_USER - OBSERVED_CV}) なら"
          f" 平均 {flat.mean():.5f} / SD {flat.std():.5f}")
    print(f"  一様事前は「成功1・失敗1を先に見た」のと同じ。平均が 1/{N_USER + 2} = "
          f"{1 / (N_USER + 2):.5f} ぶん上に寄る")
    print(f"  CVR 3% では、この {1 / (N_USER + 2):.5f} が相対で {1 / (N_USER + 2) / P_TRUE:.1%} に当たる。"
          "小さい割合ほど事前分布が効く")
    return post, flat


def compare(phat: np.ndarray, post, flat) -> None:
    print(f"\n--- {TRIALS:,} 回の p̂ の分布 vs ベータ分布 ---")
    print(f"{'突き合わせ先':<22}{'平均の差':>12}{'SDの比':>10}{'CDFの最大差':>14}")
    for name, dist in ((f"Beta({A_POST}, {B_POST})", post),
                       (f"Beta({OBSERVED_CV}, {N_USER - OBSERVED_CV})", flat)):
        print(f"{name:<22}{dist.mean() - phat.mean():>+12.5f}"
              f"{dist.std() / phat.std(ddof=1):>10.4f}{cdf_gap(phat, dist):>14.4f}")
    print("  幅（SD）はどちらもほぼ一致する。ずれているのは位置だけで、"
          "その位置のずれは事前分布1件ぶんに等しい")
    print("  ← 頻度の側の「p̂ のばらつき」と、ベイズの側の「p についての不確かさ」は、"
          "n が大きければ同じ形になる")

    print("\n--- ただし、意味は同じではない ---")
    lo_f, hi_f = np.quantile(phat, [0.025, 0.975])
    lo_b, hi_b = post.ppf(0.025), post.ppf(0.975)
    print(f"  p̂ の中央95%     [{lo_f:.4f}, {hi_f:.4f}]"
          "   ← 真の p を固定して、データを10,000回引き直した散らばり")
    print(f"  95%信用区間     [{lo_b:.4f}, {hi_b:.4f}]"
          "   ← データを1つ固定して、p についての確からしさ")
    print("  数字はほぼ同じで、動いているものが逆になっている。"
          "この区別は第6章と第15章でもう一度出てくる")


def make_figure(phat: np.ndarray, post, flat) -> None:
    plots.setup()
    fig, axes = plots.figure(1, 2, w=1.0, h=0.85, constrained_layout=True)

    grid = np.linspace(0.010, 0.055, 500)

    ax = axes[0]
    ax.hist(phat, bins=np.arange(0.0095, 0.0555, 0.001), density=True,
            color=plots.PALETTE["data"], alpha=0.55, lw=0)
    ax.plot(grid, post.pdf(grid), color=plots.PALETTE["posterior"], lw=1.4, zorder=4)
    ax.fill_between(grid, post.pdf(grid), color=plots.PALETTE["posterior"],
                    alpha=0.20, lw=0, zorder=1)
    ax.annotate(f"Beta({A_POST}, {B_POST}) の事後", xy=(0.038, post.pdf(0.038)),
                fontsize=6.0, color=plots.PALETTE["posterior"], ha="left", va="bottom",
                xytext=(2, 2), textcoords="offset points")
    ax.annotate(f"観測CVRの分布（{TRIALS:,}回）", xy=(0.0150, 12), fontsize=6.0,
                color=plots.PALETTE["data"], ha="left")
    plots.mark_truth(ax, P_TRUE, "真の p = 0.03")
    ax.set_xlabel("CVR")
    ax.set_ylabel("密度")
    ax.set_title("2つはほぼ重なる")

    ax = axes[1]
    xs = np.sort(phat)
    ax.step(xs, np.arange(1, xs.size + 1) / xs.size, where="post",
            color=plots.PALETTE["data"], lw=1.1)
    ax.plot(grid, post.cdf(grid), color=plots.PALETTE["posterior"], lw=1.3)
    ax.plot(grid, flat.cdf(grid), color=plots.PALETTE["truth"], lw=1.1, ls="--",
            dashes=(4, 2.0))
    ax.annotate("観測CVRの経験CDF", xy=(0.0230, 0.08), fontsize=6.0,
                color=plots.PALETTE["data"], ha="left")
    ax.annotate(f"Beta({A_POST}, {B_POST})\n（一様事前）", xy=(0.036, 0.72), fontsize=5.8,
                color=plots.PALETTE["posterior"], ha="left")
    ax.annotate(f"Beta({OBSERVED_CV}, {N_USER - OBSERVED_CV})", xy=(0.0155, 0.62),
                fontsize=5.8, color=plots.PALETTE["truth"], ha="left")
    ax.set_xlim(0.012, 0.052)
    ax.set_xlabel("CVR")
    ax.set_ylabel("累積確率")
    ax.set_title(f"ずれは事前1件ぶん（最大差 {cdf_gap(phat, post):.4f}）")

    plots.save(fig, "fig-4-9-beta-of-a-rate.png")


def main() -> None:
    counts = sim.repeat(one_trial, trials=TRIALS, seed=SEED, progress=False)
    phat = counts.astype(float) / N_USER

    frequentist_side(phat)
    post, flat = bayesian_side()
    compare(phat, post, flat)
    make_figure(phat, post, flat)


if __name__ == "__main__":
    main()
