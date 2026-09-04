"""確率 0.5 は「無限に投げたときの割合」— その寄り方の速さを数え上げる。

確率を「起こりやすさ」と言い換えても何も進まない。頻度主義の確率は、同じ試行を
限りなく繰り返したときに相対頻度が寄っていく先のことだった。ここでは表の確率 0.5 の
コインを 10 回から 100 万回まで投げ、相対頻度が 0.5 に寄る様子と、**寄る速さ**を測る。

要点は2つある。1つは、n=10 の 0.7 のような大きなずれが 100 万回では消えること。
もう1つは、消え方が 1/√n だということ。誤差は n に反比例しては減らない。100 倍
投げても誤差は 10 分の 1 にしかならず、この 1/√n が以降の章の標準誤差そのものになる。

    uv run python examples/ch03/frequency_probability_converges.py
"""

import numpy as np

from toukei_tashikame import plots

P_TRUE = 0.5
SEED = 6
N_MAX = 10**6


def one_long_path(seed: int) -> np.ndarray:
    """1 枚のコインを N_MAX 回投げ、各時点までの相対頻度を返す。"""
    rng = np.random.default_rng(seed)
    flips = rng.random(N_MAX) < P_TRUE
    return np.cumsum(flips) / np.arange(1, N_MAX + 1)


def rmse_at(n: int, reps: int, seed: int) -> float:
    """n 回投げを reps 本くり返し、相対頻度の二乗平均平方根誤差を測る。"""
    rng = np.random.default_rng(seed)
    heads = rng.binomial(n, P_TRUE, size=reps)
    return float(np.sqrt(np.mean((heads / n - P_TRUE) ** 2)))


def main() -> None:
    plots.setup()

    print("--- 1. 1本の道のり（seed=6）---")
    path = one_long_path(SEED)
    print(f"  {'n':>9}  {'表の割合':>10}  {'|割合 - 0.5|':>12}")
    checkpoints = [10**k for k in range(1, 7)]
    for n in checkpoints:
        r = path[n - 1]
        print(f"  {n:>9,}  {r:>10.6f}  {abs(r - P_TRUE):>12.6f}")
    # 1本だけを見ていると、小さい n でたまたま当たったのか外したのかが分からない。
    small = np.random.default_rng(SEED + 1).binomial(10, P_TRUE, size=20) / 10
    print(f"  n=10 を別の種で20本引くと  {np.round(small, 1)}")
    print(f"  ← 0.5 ちょうどから {np.abs(small - P_TRUE).max():.1f} 離れる本もある。"
          "10回では相対頻度はまだ確率ではない")

    print("\n--- 2. 寄る速さ（各 n を何本もくり返して誤差の大きさを測る）---")
    # 大きい n ほど1本が重いので本数を減らす。誤差の推定精度は本数で決まるが、
    # 傾きを見るだけならこれで足りる。
    plan = [(10, 4000), (10**2, 4000), (10**3, 2000), (10**4, 1000),
            (10**5, 400), (10**6, 100)]
    ns, rmses = [], []
    print(f"  {'n':>9}  {'本数':>5}  {'誤差(RMSE)':>11}  {'理論 √(p(1-p)/n)':>16}")
    for i, (n, reps) in enumerate(plan):
        r = rmse_at(n, reps, seed=SEED + 100 * (i + 1))
        theory = np.sqrt(P_TRUE * (1 - P_TRUE) / n)
        ns.append(n)
        rmses.append(r)
        print(f"  {n:>9,}  {reps:>5}  {r:>11.6f}  {theory:>16.6f}")

    slope, _intercept = np.polyfit(np.log10(ns), np.log10(rmses), 1)
    print(f"\n  log-log の傾き = {slope:.3f}（理論 -0.500）")
    print("  誤差 ∝ n^(-1/2)。n を 100 倍にして誤差は 10 分の 1。これが標準誤差の正体")

    fig, (ax1, ax2) = plots.figure(1, 2, w=2.0, h=0.95)

    # 左: 相対頻度の道のりと、±2/√n の帯。
    grid = np.unique(np.logspace(0, 6, 600).astype(int))
    ax1.plot(grid, path[grid - 1], color=plots.PALETTE["estimate"], lw=0.9, zorder=3)
    band = 2 * np.sqrt(P_TRUE * (1 - P_TRUE) / grid)
    ax1.fill_between(grid, np.clip(P_TRUE - band, 0, 1), np.clip(P_TRUE + band, 0, 1),
                     color=plots.PALETTE["interval"], alpha=0.18, lw=0, zorder=1)
    plots.mark_truth(ax1, P_TRUE, "真の確率 = 0.5", axis="y")
    ax1.set_xscale("log")
    ax1.set_ylim(0, 1)
    ax1.set_xlabel("投げた回数 n（対数）")
    ax1.set_ylabel("表の相対頻度")
    ax1.set_title("相対頻度は確率に寄る")
    ax1.annotate("±2/√n の帯", xy=(3e3, P_TRUE + 2 * np.sqrt(0.25 / 3e3)),
                 xytext=(2, 3), textcoords="offset points",
                 fontsize=6.0, color=plots.PALETTE["estimate"])

    # 右: 誤差の大きさそのもの。両対数で直線なら冪則。
    ax2.plot(ns, rmses, "o-", color=plots.PALETTE["estimate"], ms=3, lw=0.9, zorder=3)
    ax2.plot(ns, np.sqrt(P_TRUE * (1 - P_TRUE) / np.asarray(ns, float)),
             color=plots.PALETTE["truth"], lw=1.1, ls="--", dashes=(4, 2.0), zorder=4)
    ax2.annotate(f"傾き {slope:.3f}（理論 −0.5）", xy=(10**3, rmses[2]),
                 xytext=(4, 6), textcoords="offset points",
                 fontsize=6.0, color=plots.PALETTE["truth"])
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("投げた回数 n（対数）")
    ax2.set_ylabel("誤差の大きさ RMSE（対数）")
    ax2.set_title("寄る速さは 1/√n")

    fig.tight_layout()
    plots.save(fig, "fig-3-2-frequency-converges.png")


if __name__ == "__main__":
    main()
