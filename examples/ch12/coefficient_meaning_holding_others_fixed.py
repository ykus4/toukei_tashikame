"""重回帰の係数「他を一定にしたときの効果」を、単回帰との差として数え上げる。

真のモデルは $y = 1 + 1.0 x_1 + 1.0 x_2 + \\varepsilon$ で固定し、$x_1, x_2$ の相関だけを
0.0 / 0.5 / 0.9 と変える。$x_2$ を無視した単回帰の $\\hat b_1$ は、相関の分だけ
$x_2$ の効果を吸い込む（理論上のずれは $b_2 \\rho$）。$x_2$ を入れた重回帰の $\\hat b_1$ は
どの相関でも真値 1.0 に当たる。

ただし「当たる」ことにも値段がついている。相関が強いほど重回帰の標準誤差は膨らむ。
$x_2$ を一定にしたときに残る $x_1$ の動きが少なくなるからで、これは第12-7節の
多重共線性と同じ現象を係数の意味の側から見たものである。

    uv run python examples/ch12/coefficient_meaning_holding_others_fixed.py
"""

import numpy as np

from toukei_tashikame import plots

N, TRIALS, SEED = 500, 10_000, 123
B0, B1, B2, SIGMA = 1.0, 1.0, 1.0, 1.0
RHOS = (0.0, 0.5, 0.9)
CHUNK = 1_000        # 一度に扱う試行数。メモリを増やさずベクトル化する


def batch(rho: float, trials: int, rng):
    """``trials`` 回ぶんの単回帰・重回帰を一気に解く。

    返り値は ``(b1_単回帰, b1_重回帰, se_単回帰, se_重回帰)``。中身は中心化した
    積和から作った正規方程式そのもので、2変数なので 2x2 を手で解いている。
    """
    x1 = rng.normal(0.0, 1.0, size=(trials, N))
    x2 = rho * x1 + np.sqrt(1.0 - rho**2) * rng.normal(0.0, 1.0, size=(trials, N))
    y = B0 + B1 * x1 + B2 * x2 + rng.normal(0.0, SIGMA, size=(trials, N))

    # 切片は中心化で消える。以下はすべて中心化した積和（S11 = Σ(x1-x̄1)²）。
    x1 -= x1.mean(axis=1, keepdims=True)
    x2 -= x2.mean(axis=1, keepdims=True)
    y -= y.mean(axis=1, keepdims=True)
    s11 = (x1 * x1).sum(axis=1)
    s22 = (x2 * x2).sum(axis=1)
    s12 = (x1 * x2).sum(axis=1)
    s1y = (x1 * y).sum(axis=1)
    s2y = (x2 * y).sum(axis=1)
    syy = (y * y).sum(axis=1)

    # 単回帰（x2 を無視する）
    b1_s = s1y / s11
    sigma2_s = (syy - b1_s * s1y) / (N - 2)
    se_s = np.sqrt(sigma2_s / s11)

    # 重回帰（x2 を一定にする）
    det = s11 * s22 - s12**2
    b1_m = (s22 * s1y - s12 * s2y) / det
    b2_m = (s11 * s2y - s12 * s1y) / det
    sigma2_m = (syy - b1_m * s1y - b2_m * s2y) / (N - 3)
    se_m = np.sqrt(sigma2_m * s22 / det)
    return b1_s, b1_m, se_s, se_m


def main() -> None:
    plots.setup()
    rng = np.random.default_rng(SEED)

    print(f"--- 真の b1 = {B1}、b2 = {B2}、n={N}、{TRIALS:,} 回ずつ ---")
    print("   x1,x2 の相関   単回帰の b1     重回帰の b1     単回帰のずれ   se比(重/単)")
    results = {}
    for rho in RHOS:
        parts = [batch(rho, min(CHUNK, TRIALS - i), rng)
                 for i in range(0, TRIALS, CHUNK)]
        b1_s, b1_m, se_s, se_m = (np.concatenate([p[j] for p in parts]) for j in range(4))
        results[rho] = (b1_s, b1_m)
        print(f"      {rho:.1f}         {b1_s.mean():.4f}          {b1_m.mean():.4f}"
              f"        {b1_s.mean() - B1:+.4f}        {se_m.mean() / se_s.mean():.2f} 倍")
        print(f"                 （理論上のずれ b2×ρ = {B2 * rho:+.2f}）"
              f"  標準偏差 {b1_s.std(ddof=1):.4f} / {b1_m.std(ddof=1):.4f}")

    print("\n  単回帰の b1 は「x2 も一緒に動いたときの x1 の効果」を測っている。")
    print("  重回帰の b1 は「x2 を止めたときの x1 の効果」で、こちらが真値に当たる。")
    print("  どちらが正しいかではなく、答えている問いが違う。")
    print("  相関 0 のときは x2 を入れたほうが誤差分散が減るので se は小さくなる（0.71倍）。")
    print("  相関 0.9 では逆に倍以上に膨らむ——x2 を止めると x1 の動く余地が残らないため。")

    # --- 図 ---
    fig, axes = plots.figure(1, 3, w=1.9, h=0.95, sharey=True)
    bins = np.linspace(0.6, 2.2, 80)
    for ax, rho in zip(axes, RHOS, strict=True):
        b1_s, b1_m = results[rho]
        ax.hist(b1_s, bins=bins, color=plots.PALETTE["data"], alpha=0.55, lw=0)
        ax.hist(b1_m, bins=bins, color=plots.PALETTE["estimate"], alpha=0.65, lw=0)
        plots.mark_truth(ax, B1, f"真値 {B1}")
        ax.annotate(f"単回帰 {b1_s.mean():.3f}", xy=(b1_s.mean(), 0.60),
                    xycoords=("data", "axes fraction"), xytext=(3, 0),
                    textcoords="offset points", fontsize=6.0, color=plots.PALETTE["data"])
        ax.annotate(f"重回帰 {b1_m.mean():.3f}", xy=(b1_m.mean(), 0.80),
                    xycoords=("data", "axes fraction"), xytext=(-3, 0), ha="right",
                    textcoords="offset points", fontsize=6.0,
                    color=plots.PALETTE["estimate"])
        ax.set_title(f"$x_1, x_2$ の相関 = {rho:.1f}")
        ax.set_xlabel("$\\hat{b}_1$")
    axes[0].set_ylabel(f"{TRIALS:,} 回のうちの回数")
    fig.tight_layout()
    plots.save(fig, "fig-12-3-simple-vs-multiple-coef.png")


if __name__ == "__main__":
    main()
