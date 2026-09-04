"""検出力曲線 — n を増やすと検出力はどう上がるか。d=0.2 / 0.5 / 0.8 で描く。

「n を2倍にすれば検出力も2倍」ではない。曲線は S 字で、下のほうは伸びが鈍く、0.8 を
超えたあたりから急に伸びなくなる。だから「あと少し足りない」ときほど追加コストが高い。
小さい効果ほど曲線は右に寝る——d=0.2 と d=0.8 で必要な n は25倍違う。

各点を5,000回ずつ数え上げ、非心 t 分布の理論曲線を重ねる。数え上げは1点ずつ独立に
回すので、点が曲線の上下にばらつくのは実装の誤りではなくシミュレーション誤差である。

    uv run python examples/ch10/power_curve_vs_n.py
"""

import numpy as np
from scipy import stats

from toukei_tashikame import plots, power

DS = (0.2, 0.5, 0.8)
N_SIM = (5, 10, 20, 40, 80, 150, 300)       # 数え上げる点
TRIALS, ALPHA, SEED = 5_000, 0.05, 103


def simulate_power(n: int, d: float, trials: int, rng) -> float:
    """(trials, n) の行列を2枚引いて、Welch の t 検定を一気にかける。

    1回ずつ回しても同じ結果になるが、曲線は点が多く、まとめて引かないと待たされる。
    """
    a = rng.normal(0.0, 1.0, size=(trials, n))
    b = rng.normal(d, 1.0, size=(trials, n))
    va, vb = a.var(axis=1, ddof=1), b.var(axis=1, ddof=1)
    se = np.sqrt(va / n + vb / n)
    t = (b.mean(axis=1) - a.mean(axis=1)) / se
    df = se**4 / ((va / n) ** 2 / (n - 1) + (vb / n) ** 2 / (n - 1))
    p = 2 * stats.t.sf(np.abs(t), df)
    return float((p < ALPHA).mean())


def main() -> None:
    plots.setup()
    rng = np.random.default_rng(SEED)

    print(f"--- 検出力曲線（α={ALPHA}、各点 {TRIALS:,} 回）---")
    print("      n  " + "".join(f"    d={d}（実測/理論）" for d in DS))
    table = {d: [simulate_power(n, d, TRIALS, rng) for n in N_SIM] for d in DS}
    for i, n in enumerate(N_SIM):
        cells = "".join(f"      {table[d][i]:.3f} / {power.power_ttest(n, d, ALPHA):.3f}"
                        for d in DS)
        print(f"  {n:5d}{cells}")

    print("\n--- 検出力 0.80 に届く最小の n（各群、二分探索）---")
    needed = {d: power.n_for_power(d, power=0.8, alpha=ALPHA) for d in DS}
    for d in DS:
        print(f"  d={d}  n = {needed[d]:5d} / 群（総数 {2 * needed[d]:,}）"
              f"  実測 {simulate_power(needed[d], d, TRIALS, rng):.4f}")
    print(f"\n  効果量が {DS[-1]}→{DS[0]} と {DS[-1] / DS[0]:.0f} 分の1になると、"
          f"必要な n は {needed[DS[0]] / needed[DS[-1]]:.1f} 倍になった"
          f"（2乗に反比例するなら {(DS[-1] / DS[0]) ** 2:.0f} 倍）。")
    print("  n は効果量の2乗に反比例する。小さい効果を見に行くのが高くつくのはこのため。")

    # --- 図 ---
    fig, ax = plots.figure(w=1.15, h=1.15)
    grid = np.unique(np.round(np.logspace(np.log10(5), np.log10(300), 60)).astype(int))
    for d in DS:
        ax.plot(grid, [power.power_ttest(int(n), d, ALPHA) for n in grid],
                color=plots.PALETTE["estimate"], lw=1.1, zorder=3)

        ax.scatter(N_SIM, table[d], s=9, color=plots.PALETTE["data"], lw=0, zorder=4)
        # ラベルは曲線が 0.5 を横切るあたりに置く（右端に並べると重なる）
        curve = np.array([power.power_ttest(int(n), d, ALPHA) for n in grid])
        j = int(np.argmin(np.abs(curve - 0.55)))
        ax.annotate(f"d = {d}", xy=(grid[j], curve[j]), xytext=(-3, 6),
                    textcoords="offset points", ha="right",
                    fontsize=6.0, color=plots.PALETTE["estimate"])
    ax.axhline(0.8, color=plots.PALETTE["truth"], lw=1.0, zorder=5)
    ax.annotate("目標 0.80", xy=(5.5, 0.81), fontsize=6.0, color=plots.PALETTE["truth"])
    for d in DS:
        if needed[d] <= 300:
            ax.plot([needed[d], needed[d]], [0, 0.8], color=plots.PALETTE["reject"],
                    lw=0.8, ls="--", dashes=(3, 2), zorder=2)
            ax.annotate(f"n={needed[d]}", xy=(needed[d], 0.03), fontsize=6.0,
                        ha="center", color=plots.PALETTE["reject"])
    ax.set_xscale("log")
    ax.set_xlabel("n（各群、対数目盛）")
    ax.set_ylabel("検出力")
    ax.set_ylim(0, 1.02)
    ax.set_title("点が数え上げ、線が非心 t の理論値")
    fig.tight_layout()
    plots.save(fig, "fig-10-3-power-curve.png")


if __name__ == "__main__":
    main()
