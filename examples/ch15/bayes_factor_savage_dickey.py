"""ベイズファクターを密度の比で手計算し、事前の幅で答えが動くことを見る。

Savage–Dickey の密度比は、入れ子になった仮説なら周辺尤度の積分を解かずに
$BF_{10} = \\pi(0) / \\pi(0 \\mid \\text{data})$ で済ませられる、という定理である。
「帰無の点における事前密度を事後密度で割る」だけで、割り算1回で出る。

その代わり、事前分布の幅 $\\tau$ が答えに直接残る。$\\tau$ を狭くすれば $H_1$ が $H_0$ に
近づいて $BF \\to 1$、広げすぎれば $H_1$ が薄まって $BF \\to 0$ に向かう（Jeffreys–Lindley）。
p 値は同じ 1 つの値のままなのに、ベイズファクターは事前の幅で 1.1 から 2.9 まで動く。
**ベイズファクターを報告するなら事前を必ず書く**、の根拠がここにある。

    uv run python examples/ch15/bayes_factor_savage_dickey.py
"""

import numpy as np
from scipy import stats

from toukei_tashikame import plots

N, SIGMA, TRUE_EFFECT, SEED = 30, 1.0, 0.70, 158
TAUS = (0.05, 0.10, 0.25, 0.50, 0.75, 1.00, 2.00, 5.00)


def savage_dickey(xbar: float, tau: float) -> tuple[float, float, float, float]:
    """``(BF10, 事前密度(0), 事後密度(0), 事後平均)``。

    σ 既知の正規モデルなら事後も正規で、分散は精度の足し算で出る。
    事前 N(0, τ²) と尤度 N(x̄, σ²/n) の精度を足して逆数を取るだけ。
    """
    prec = 1.0 / tau**2 + N / SIGMA**2          # 事前の精度 + データの精度
    var_post = 1.0 / prec
    mean_post = var_post * (N * xbar / SIGMA**2)
    d_prior = stats.norm.pdf(0.0, 0.0, tau)                    # π(0)
    d_post = stats.norm.pdf(0.0, mean_post, np.sqrt(var_post))  # π(0 | data)
    return d_prior / d_post, d_prior, d_post, mean_post


def marginal_ratio(xbar: float, tau: float) -> float:
    """答え合わせ用。周辺尤度を直接比べる（σ 既知なら解析的に書ける）。"""
    se2 = SIGMA**2 / N
    m1 = stats.norm.pdf(xbar, 0.0, np.sqrt(se2 + tau**2))   # H1: μ ~ N(0, τ²)
    m0 = stats.norm.pdf(xbar, 0.0, np.sqrt(se2))            # H0: μ = 0
    return float(m1 / m0)


def draw(xbar: float, taus: np.ndarray, bfs: np.ndarray, pvalue: float) -> None:
    fig, axes = plots.figure(1, 2, w=2.0, h=0.95)
    pal = plots.PALETTE
    tau0 = 0.50

    ax = axes[0]
    bf0, d_prior, d_post, mean_post = savage_dickey(xbar, tau0)
    grid = np.linspace(-1.5, 1.5, 800)
    var_post = 1.0 / (1.0 / tau0**2 + N / SIGMA**2)
    plots.prior_posterior(ax, grid,
                          prior=stats.norm.pdf(grid, 0.0, tau0),
                          posterior=stats.norm.pdf(grid, mean_post, np.sqrt(var_post)))
    ax.plot([0.0, 0.0], [d_post, d_prior], color=pal["reject"], lw=1.4, zorder=6)
    ax.plot([0.0, 0.0], [d_post, d_prior], ls="none", marker="o", ms=3.0,
            color=pal["reject"], zorder=7)
    ax.annotate(f"事前 $\\pi(0) = {d_prior:.3f}$", xy=(0.0, d_prior), xytext=(-6, 4),
                textcoords="offset points", ha="right", fontsize=6.0, color=pal["prior"])
    ax.annotate(f"事後 $\\pi(0 \\mid D) = {d_post:.3f}$", xy=(0.0, d_post), xytext=(-6, -9),
                textcoords="offset points", ha="right", fontsize=6.0, color=pal["posterior"])
    ax.annotate(f"比 = $BF_{{10}}$ = {bf0:.2f}", xy=(0.05, 0.55), xycoords="axes fraction",
                fontsize=6.2, color=pal["reject"])
    ax.set_xlabel("$\\mu$")
    ax.set_ylabel("密度")
    ax.set_title(f"① 0 における密度の比（事前SD = {tau0}）")

    ax = axes[1]
    ax.plot(taus, bfs, color=pal["posterior"], lw=1.3, zorder=4)
    ax.axhline(1.0, color=pal["ink2"], lw=0.7, zorder=3)
    ax.annotate("$BF=1$（どちらとも言えない）", xy=(taus[0], 1.0), xytext=(2, 3),
                textcoords="offset points", fontsize=5.8, color=pal["ink2"])
    ax.axhline(3.0, color=pal["truth"], lw=1.0, ls="--", dashes=(4, 2.0), zorder=3)
    ax.annotate("$BF=3$（弱い証拠の目安）", xy=(taus[0], 3.0), xytext=(2, -9),
                textcoords="offset points", fontsize=5.8, color=pal["truth"])
    best = int(np.argmax(bfs))
    ax.plot([taus[best]], [bfs[best]], marker="o", ms=3.2, color=pal["reject"], zorder=6)
    ax.annotate(f"最大 {bfs[best]:.2f} @ $\\tau$={taus[best]:.2f}", xy=(taus[best], bfs[best]),
                xytext=(8, -10), textcoords="offset points", fontsize=6.0, color=pal["reject"])
    ax.annotate(f"p 値はこの間ずっと {pvalue:.4f} のまま", xy=(0.30, 0.10),
                xycoords="axes fraction", fontsize=6.0, color=pal["ink2"])
    ax.set_xscale("log")
    ax.set_xlabel("事前分布の SD $\\tau$（対数）")
    ax.set_ylabel("$BF_{10}$")
    ax.set_title("② 同じデータでも事前の幅で BF は動く")

    plots.save(fig, "fig-15-8-bayes-factor-vs-prior-width.png")


def main() -> None:
    plots.setup()
    x = np.random.default_rng(SEED).normal(TRUE_EFFECT, SIGMA, size=N)
    xbar = float(x.mean())
    se = SIGMA / np.sqrt(N)
    z = xbar / se
    pvalue = float(2 * stats.norm.sf(abs(z)))

    print(f"--- n={N}、σ={SIGMA}（既知）、真の効果 {TRUE_EFFECT}、seed={SEED} ---\n")
    print(f"  標本平均 x̄ = {xbar:.4f}、SE = {se:.4f}")
    print(f"  頻度論の両側 z 検定    z = {z:.4f}、p = {pvalue:.4f}"
          f"   → α=0.05 で{'棄却する' if pvalue < 0.05 else '棄却しない'}")
    print("  ここから先、データは一切変わらない。動かすのは事前分布の幅だけである。\n")

    print("  Savage–Dickey: BF10 = 事前密度(0) ÷ 事後密度(0)。積分は要らない。\n")
    print(f"{'事前SD τ':>10}{'π(0)':>10}{'π(0|D)':>10}{'事後平均':>11}"
          f"{'BF10':>9}{'周辺尤度比':>12}{'読み':>16}")
    taus = np.array(TAUS)
    bfs = []
    for tau in taus:
        bf, d_prior, d_post, mean_post = savage_dickey(xbar, float(tau))
        bfs.append(bf)
        if bf < 1:
            verdict = "H0 寄り"
        elif bf < 3:
            verdict = "ほぼ無情報"
        elif bf < 10:
            verdict = "H1 の弱い証拠"
        else:
            verdict = "H1 の強い証拠"
        print(f"{tau:>10.2f}{d_prior:>10.4f}{d_post:>10.4f}{mean_post:>11.4f}"
              f"{bf:>9.3f}{marginal_ratio(xbar, float(tau)):>12.3f}{verdict:>16}")
    bfs = np.array(bfs)

    diff = float(np.abs(bfs - [marginal_ratio(xbar, float(t)) for t in taus]).max())
    print(f"\n  Savage–Dickey と周辺尤度比の最大差 {diff:.1e}"
          "  ← 定理どおり、割り算1回で積分と同じ答えになる\n")

    lo, mid, hi = savage_dickey(xbar, 0.05)[0], savage_dickey(xbar, 0.50)[0], \
        savage_dickey(xbar, 1.00)[0]
    print(f"  事前SD 0.05 で BF={lo:.2f}、0.50 で BF={mid:.2f}、1.00 で BF={hi:.2f}。")
    print("  単調ではない。狭すぎる事前は H1 を H0 とほとんど同じにしてしまい（BF→1）、")
    print("  広すぎる事前は H1 の確率を「ありえない大きさの効果」に配りきってしまう。")
    print("  後者が Jeffreys–Lindley のパラドックスで、τ を無限に広げると、p 値が")
    print("  いくら小さくても BF は 0 に向かう。表の τ=5.00 の行がその途中である。\n")

    fine = np.logspace(np.log10(0.02), np.log10(20.0), 400)
    fine_bf = np.array([savage_dickey(xbar, float(t))[0] for t in fine])
    best = int(np.argmax(fine_bf))
    print(f"  細かく刻むと BF は τ = {fine[best]:.3f} で最大 {fine_bf[best]:.3f} を取り、")
    print(f"  τ = {fine[fine_bf < 1][0] if (fine_bf < 1).any() else float('nan'):.2f} "
          "を超えると 1 を割って H0 寄りに転じる。\n")

    print(f"p 値 {pvalue:.4f} は「有意」と読める1つの数字だが、同じデータのベイズファクターは")
    print(f"事前の書き方次第で {bfs.min():.2f} から {bfs.max():.2f} まで動いた。")
    print("どちらが正しいかではなく、BF は「H1 として何を想定したか」まで含めた量である。")
    print("報告するときは BF だけでなく、事前分布とその幅を必ず添える。")
    draw(xbar, fine, fine_bf, pvalue)


if __name__ == "__main__":
    main()
