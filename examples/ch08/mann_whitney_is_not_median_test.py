"""崩し — Mann-Whitney の U 検定は「中央値の検定」ではない。

U 検定はしばしば「ノンパラメトリック版の t 検定」「中央値を比べる検定」と紹介される。
実際に見ているのは P(X > Y) が 1/2 かどうかであって、中央値ではない。分布の形が違えば、
**中央値が完全に一致していても**棄却される。

中央値がどちらもちょうど 1 になる2分布——対数正規 exp(N(0,1)) と正規 N(1,1)——から
n=50 ずつ引いて 10,000 回検定する。「位置のずれ」と読めるのは、2群の分布の形が同じ
ときだけである。

    uv run python examples/ch08/mann_whitney_is_not_median_test.py
"""

import numpy as np
from scipy import stats

from toukei_tashikame import plots, testing

N, TRIALS, SEED, ALPHA = 50, 10_000, 86, 0.05
MEDIAN = 1.0   # 両分布に共通の中央値


def draw_a(rng, shape):
    """対数正規 exp(N(0,1))。中央値 = exp(0) = 1、平均 = exp(0.5) ≈ 1.6487。"""
    return rng.lognormal(0.0, 1.0, size=shape)


def draw_b(rng, shape):
    """正規 N(1,1)。中央値 = 平均 = 1。"""
    return rng.normal(MEDIAN, 1.0, size=shape)


def se(rate: float, trials: int = TRIALS) -> float:
    return float(np.sqrt(rate * (1 - rate) / trials))


def main() -> None:
    plots.setup()

    # 1. 母集団の性質。中央値は一致し、P(X>Y) は 1/2 から外れる
    big = np.random.default_rng(SEED + 1)
    xa, xb = draw_a(big, 400_000), draw_b(big, 400_000)
    p_xy = float((xa > xb).mean())
    print("--- 母集団（真の値。合成データだけが知っている）---")
    print(f"  A: 対数正規 exp(N(0,1))   中央値 = {np.median(xa):.4f}  平均 = {xa.mean():.4f}")
    print(f"  B: 正規 N(1,1)            中央値 = {np.median(xb):.4f}  平均 = {xb.mean():.4f}")
    print(f"  中央値の差 = {np.median(xa) - np.median(xb):+.4f}   ← 帰無「中央値は等しい」は真")
    print(f"  P(X > Y)   = {p_xy:.4f}   ← U 検定が見ているのはこちら。1/2 から外れている")

    # 2. 数え上げ
    rng = np.random.default_rng(SEED)
    a = draw_a(rng, (TRIALS, N))
    b = draw_b(rng, (TRIALS, N))
    p_u = stats.mannwhitneyu(a, b, axis=1).pvalue
    p_t = stats.ttest_ind(a, b, axis=1, equal_var=False).pvalue
    p_med = np.array([stats.median_test(x, y)[1] for x, y in zip(a, b, strict=True)])

    rows = [
        ("Mann-Whitney の U 検定", float((p_u < ALPHA).mean())),
        ("Mood の中央値検定", float((p_med < ALPHA).mean())),
        ("Welch の t 検定（参考）", float((p_t < ALPHA).mean())),
    ]
    print(f"\n--- 棄却率（n={N} ずつ、{TRIALS:,}回、α={ALPHA}）---")
    for label, v in rows:
        print(f"  {label:<26} {v:.4f} ± {1.96 * se(v):.4f}")

    print(f"\n  中央値は完全に一致しているのに、U 検定は {rows[0][1]:.4f}"
          f"（名目の {rows[0][1] / ALPHA:.1f}倍）で棄却する")
    print(f"  中央値そのものを見る Mood の検定は {rows[1][1]:.4f} で名目を超えない"
          "（連続修正のぶん保守的）。U 検定が拾っているのは中央値の差ではない")
    print(f"  参考の t 検定が {rows[2][1]:.4f} なのは、平均が "
          f"{xa.mean():.4f} と {xb.mean():.4f} で本当に違うから。こちらは誤りではない")
    print("  U 検定を「位置のずれ」と読めるのは、2群の分布の**形が同じ**ときだけである")

    # 3. 手書き実装との照合（1組ぶん）
    hand = testing.mann_whitney_u(a[0], b[0])
    lib = stats.mannwhitneyu(a[0], b[0])
    print(f"\n--- 1組ぶんの照合 ---\n  {hand}")
    print(f"  scipy の U = {lib.statistic:.1f}, p = {lib.pvalue:.4g}"
          f"（手書きとの差 {abs(hand.pvalue - lib.pvalue):.2e}）")

    # --- 図 ---
    fig, (ax1, ax2) = plots.figure(1, 2, w=1.6)

    grid = np.linspace(-2.5, 6.0, 400)
    ax1.plot(grid, stats.lognorm.pdf(grid, s=1.0), color=plots.PALETTE["alt"], lw=1.3)
    ax1.plot(grid, stats.norm.pdf(grid, MEDIAN, 1.0), color=plots.PALETTE["data"], lw=1.3)
    ax1.annotate("A: 対数正規", xy=(0.42, stats.lognorm.pdf(0.42, s=1.0)), xytext=(-4, 0),
                 textcoords="offset points", ha="right", va="center", fontsize=6.0,
                 color=plots.PALETTE["alt"])
    ax1.annotate("B: 正規", xy=(2.2, stats.norm.pdf(2.2, MEDIAN, 1.0)), xytext=(4, 2),
                 textcoords="offset points", fontsize=6.0, color=plots.PALETTE["data"])
    plots.mark_truth(ax1, MEDIAN, "共通の中央値 = 1")
    ax1.set_xlabel("値")
    ax1.set_ylabel("密度")
    ax1.set_title(f"中央値は同じ、形は違う（P(X>Y)={p_xy:.3f}）")

    values = [v for _, v in rows]
    bars = ax2.bar([0, 1, 2], values, width=0.5,
                   color=[plots.PALETTE["reject"], plots.PALETTE["estimate"],
                          plots.PALETTE["data"]], alpha=0.85, lw=0)
    for rect, v in zip(bars, values, strict=True):
        ax2.annotate(f"{v:.4f}", xy=(rect.get_x() + rect.get_width() / 2, v),
                     xytext=(0, 2), textcoords="offset points", ha="center", fontsize=6.5)
    plots.mark_truth(ax2, ALPHA, "名目 α = 0.05", axis="y")
    ax2.set_xticks([0, 1, 2])
    ax2.set_xticklabels(["U 検定", "中央値検定", "t 検定\n（参考）"])
    ax2.set_ylim(0, max(values) * 1.2)
    ax2.set_ylabel("棄却率")
    ax2.set_title(f"「中央値は等しい」が真のときの棄却率（{TRIALS:,}回）")

    fig.tight_layout()
    plots.save(fig, "fig-8-6-mannwhitney-not-median.png")


if __name__ == "__main__":
    main()
