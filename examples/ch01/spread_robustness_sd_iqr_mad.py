"""汚染を 0% から 10% まで上げると、標準偏差だけが壊れる。

N(0,1) に N(0,100)（標準偏差 10 の広い正規）を少しずつ混ぜる。混ぜても「本体」の
ばらつきは 1 のままなので、ばらつきの推定量はどれも 1 を返してほしい。

標準偏差は 2 乗を平均するので、遠い値が 2 乗で効いて 3 倍以上に膨らむ。IQR と MAD は
順位しか見ないので、混入率が数 % なら 1 の近くに留まる。どちらも正規分布のもとで
標準偏差と同じ目盛りになるよう定数（IQR は ÷1.349、MAD は ×1.4826）で揃えてある。

    uv run python examples/ch01/spread_robustness_sd_iqr_mad.py
"""

import numpy as np

from toukei_tashikame import datasets, describe, plots

N = 1_000
SEED = 2
IQR_SCALE = 1.349      # 正規分布での IQR = 1.349σ
EPS_GRID = np.array([0.00, 0.02, 0.04, 0.06, 0.08, 0.10])


def spreads(x) -> tuple[float, float, float]:
    """3つの尺度を、どれも「正規分布での σ」の目盛りに揃えて返す。"""
    return describe.sd(x), describe.iqr(x) / IQR_SCALE, describe.mad(x)


def main() -> None:
    plots.setup()

    rows = []
    for eps in EPS_GRID:
        # 汚染は N(0,1) と N(0,10²) の混合。外れ値を後から足すのではなく最初から混ぜる。
        x = datasets.contaminated(N, eps=float(eps), scale=10.0, seed=SEED)
        rows.append(spreads(x))
    sd, iqr_s, mad_s = np.array(rows).T

    print(f"--- N(0,1) に N(0,100) を混ぜる（n={N:,}, seed={SEED}）---")
    print(f"{'汚染率':>8}{'SD':>10}{'IQR/1.349':>12}{'MAD×1.4826':>12}{'期待混入数':>12}")
    for eps, a, b, c in zip(EPS_GRID, sd, iqr_s, mad_s, strict=True):
        n_expected = round(float(eps) * N)   # 実際の混入数は二項乱数なのでこの前後で揺れる
        print(f"{eps:>7.0%}{a:>10.4f}{b:>12.4f}{c:>12.4f}{n_expected:>12d}")

    print(f"\n  0% → 10% での増え方   SD {sd[0]:.4f} → {sd[-1]:.4f}"
          f"（{100 * (sd[-1] / sd[0] - 1):+.1f}%）")
    print(f"{'':<22}IQR基準 {iqr_s[0]:.4f} → {iqr_s[-1]:.4f}"
          f"（{100 * (iqr_s[-1] / iqr_s[0] - 1):+.1f}%）")
    print(f"{'':<22}MAD基準 {mad_s[0]:.4f} → {mad_s[-1]:.4f}"
          f"（{100 * (mad_s[-1] / mad_s[0] - 1):+.1f}%）")
    print("  混合分布の「本当の」標準偏差は汚染率とともに実際に増えている（10%なら √(0.9+0.1×100)≒3.30）。")
    print("  SD はそれを正しく測っている。壊れているのではなく、"
          "『本体のばらつき』を知りたいときに答えてくれないだけである")

    fig, ax = plots.figure()
    ax.plot(EPS_GRID, sd, color=plots.PALETTE["reject"], lw=1.3, marker="o", ms=3)
    ax.plot(EPS_GRID, iqr_s, color=plots.PALETTE["estimate"], lw=1.3, marker="s", ms=3)
    ax.plot(EPS_GRID, mad_s, color=plots.PALETTE["alt"], lw=1.3, marker="^", ms=3,
            ls="--", dashes=(4, 2.0))
    plots.mark_truth(ax, 1.0, "本体のばらつき = 1", axis="y")
    ax.annotate("SD", xy=(EPS_GRID[-1], sd[-1]), fontsize=6.4, color=plots.PALETTE["reject"],
                ha="right", va="bottom")
    ax.annotate("IQR/1.349", xy=(EPS_GRID[2], iqr_s[2]), fontsize=6.4,
                color=plots.PALETTE["estimate"], ha="center", va="bottom",
                xytext=(0, 12), textcoords="offset points")
    ax.annotate("MAD×1.4826", xy=(EPS_GRID[4], mad_s[4]), fontsize=6.4,
                color=plots.PALETTE["alt"], ha="center", va="top",
                xytext=(0, -8), textcoords="offset points")
    ax.set_xlabel("汚染率")
    ax.set_ylabel("ばらつきの推定値")
    ax.set_title("汚染率を上げたときの3つの尺度")

    plots.save(fig, "fig-1-3-robust-spread.png")


if __name__ == "__main__":
    main()
