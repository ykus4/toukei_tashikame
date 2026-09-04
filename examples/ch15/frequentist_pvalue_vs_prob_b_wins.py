"""同じA/Bデータに検定とベイズを両方当て、答えている問いが違うことを見る。

p 値は「差がないと仮定したとき、この差以上が出る確率」で、主語は**データ**である。
一方ベイズが返す $\\Pr[p_B > p_A]$ は「B のほうが良い確率」で、主語は**パラメータ**に
なる。前者は 0.4 でも後者は 0.7 を超える、ということが平気で起きる。矛盾ではなく、
別の問いに答えているからそうなる。

意思決定にそのまま使えるのは後者の形をしているが、後者は事前分布を1つ選んだ上での
数字である。どちらが上ということではなく、何を聞いたかを覚えておく必要がある。

    uv run python examples/ch15/frequentist_pvalue_vs_prob_b_wins.py
"""

import numpy as np
from scipy import stats

from toukei_tashikame import bayes, datasets, plots, testing

N_A, N_B = 2000, 2000
P_A, LIFT, SEED = 0.030, 0.10, 15
PRIOR_A, PRIOR_B = 1.0, 1.0      # Beta(1,1) = 一様。「何も見ていない」に対応する
DRAWS = 200_000


def frequentist(k_a: int, k_b: int) -> tuple[float, float, tuple[float, float]]:
    """比率の差の検定と、差の 95% 信頼区間（Wald）。"""
    res = testing.prop_2samp(k_b, N_B, k_a, N_A, method="score")
    p_a_hat, p_b_hat = k_a / N_A, k_b / N_B
    se = np.sqrt(p_a_hat * (1 - p_a_hat) / N_A + p_b_hat * (1 - p_b_hat) / N_B)
    half = stats.norm.ppf(0.975) * se
    diff = p_b_hat - p_a_hat
    return diff, res.pvalue, (diff - half, diff + half)


def draw(post_a, post_b, diff_samples: np.ndarray, prob_b: float) -> None:
    fig, axes = plots.figure(1, 2, w=2.0, h=0.95)
    pal = plots.PALETTE

    ax = axes[0]
    grid = np.linspace(0.015, 0.045, 600)
    for post, color, label in ((post_a, "data", "A の事後"), (post_b, "posterior", "B の事後")):
        y = post.pdf(grid)
        ax.plot(grid, y, color=pal[color], lw=1.3, zorder=4)
        ax.fill_between(grid, y, color=pal[color], alpha=0.18, lw=0, zorder=1)
        ax.annotate(label, xy=(post.mean, post.pdf(post.mean)), xytext=(3, 4),
                    textcoords="offset points", fontsize=6.0, color=pal[color])
    plots.mark_truth(ax, P_A * (1 + LIFT), f"B の真値 = {P_A * (1 + LIFT):.3f}")
    ax.set_xlabel("CVR")
    ax.set_ylabel("事後密度")
    ax.set_title("① 2つの事後分布は大きく重なっている")

    ax = axes[1]
    counts, edges = np.histogram(diff_samples, bins=70, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    ax.plot(centers, counts, color=pal["posterior"], lw=1.2, zorder=4)
    ax.fill_between(centers, counts, where=centers > 0, color=pal["posterior"],
                    alpha=0.35, lw=0, zorder=2)
    ax.fill_between(centers, counts, where=centers <= 0, color=pal["data"],
                    alpha=0.20, lw=0, zorder=2)
    ax.axvline(0.0, color=pal["ink2"], lw=0.8, zorder=5)
    ax.annotate(f"$\\Pr[p_B > p_A] = {prob_b:.4f}$\n（青く塗った面積）",
                xy=(0.60, 0.80), xycoords="axes fraction", fontsize=6.2,
                color=pal["estimate"], ha="left", va="top")
    ax.set_xlabel("$p_B - p_A$ の事後")
    ax.set_ylabel("事後密度")
    ax.set_title("② 差の事後。0 より右の面積が「B が勝つ確率」")

    plots.save(fig, "fig-15-1-pvalue-vs-prob-b-wins.png")


def main() -> None:
    plots.setup()
    d = datasets.ab_test(n_a=N_A, n_b=N_B, p_a=P_A, lift=LIFT, seed=SEED)
    k_a, k_b = int(d.a.sum()), int(d.b.sum())

    diff, pvalue, (lo, hi) = frequentist(k_a, k_b)
    post_a = bayes.beta_binomial(k_a, N_A, PRIOR_A, PRIOR_B)
    post_b = bayes.beta_binomial(k_b, N_B, PRIOR_A, PRIOR_B)
    prob_b = bayes.prob_b_beats_a(post_a, post_b, draws=DRAWS, seed=SEED)
    loss_a, loss_b = bayes.expected_loss(post_a, post_b, draws=DRAWS, seed=SEED)

    rng = np.random.default_rng(SEED)
    diff_samples = (rng.beta(post_b.a, post_b.b, DRAWS)
                    - rng.beta(post_a.a, post_a.b, DRAWS))
    d_lo, d_hi = bayes.credible_interval(diff_samples, 0.95)

    print(f"--- 同一データ（A: {k_a}/{N_A}、B: {k_b}/{N_B}、真値は {d.p_a:.3f} と "
          f"{d.p_b:.4f}）---\n")
    print("① 頻度論 — 主語はデータ")
    print(f"  観測 CVR          A {k_a / N_A:.4f}   B {k_b / N_B:.4f}")
    print(f"  比率差            {100 * diff:+.2f}pt")
    print(f"  p 値              {pvalue:.4f}")
    print(f"  差の 95%信頼区間  [{100 * lo:+.2f}pt, {100 * hi:+.2f}pt]")
    print("  読み方: 「差がないとしても、この程度の差はよく出る」。0 を跨いでいるので")
    print("          有意ではない。ただし「差がない」と言ったわけではない\n")

    print("② ベイズ — 主語はパラメータ（事前は Beta(1,1)）")
    print(f"  A の事後  {post_a}")
    print(f"  B の事後  {post_b}")
    print(f"  差の 95%信用区間  [{100 * d_lo:+.2f}pt, {100 * d_hi:+.2f}pt]")
    print(f"  Pr[p_B > p_A]     {prob_b:.4f}")
    print(f"  期待損失          B を選んで外したとき {100 * loss_b:.4f}pt / "
          f"A に留まって外したとき {100 * loss_a:.4f}pt")
    print(f"  読み方: 「B のほうが良い確率が {prob_b:.1%}」。切り替えて外したときに失う")
    print(f"          CVR は平均 {100 * loss_b:.4f}pt しかない、まで一息に言える\n")

    print(f"p 値 {pvalue:.4f}（有意でない）と Pr[p_B > p_A] = {prob_b:.4f} は矛盾していない。")
    print("前者は帰無仮説のもとでのデータの確率、後者はデータのもとでのパラメータの確率で、")
    print("条件づけの向きが逆である。同じ数字を期待するほうが間違っている。")
    print("そして後者は Beta(1,1) を選んだ上での数字であることを、必ず添えて報告する。")
    draw(post_a, post_b, diff_samples, prob_b)


if __name__ == "__main__":
    main()
