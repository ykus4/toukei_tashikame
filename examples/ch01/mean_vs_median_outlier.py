"""外れ値を1点足すと、平均だけが動いて中央値は動かない。

年収データ 200 人分に、5億円（50,000万円）の1人を足す。標本は 201 個になっただけで、
199 人の年収は 1 円も変わっていない。それでも平均は 200 万円以上跳ね、中央値はほとんど
動かない。平均は全員の値を足し合わせるので、1点の暴走がそのまま n 分の 1 で効く。
中央値は「順番の真ん中」しか見ないので、上端に何を足しても順位が 1 つずれるだけである。

「頑健（ロバスト）」という言葉の中身は、この 2 つの数字の動きの差そのものだ。

    uv run python examples/ch01/mean_vs_median_outlier.py
"""

import numpy as np

from toukei_tashikame import datasets, describe, plots

OUTLIER = 50_000.0   # 万円。5億円プレイヤーが1人だけ混ざる


def main() -> None:
    plots.setup()

    x = datasets.income(n=200, seed=0)
    x_out = np.append(x, OUTLIER)

    before = {"平均": describe.mean(x), "中央値": describe.median(x)}
    after = {"平均": describe.mean(x_out), "中央値": describe.median(x_out)}

    print(f"--- 年収データ n={x.size} に {OUTLIER:,.0f}万円 を1点足す ---")
    print(f"{'統計量':<8}{'外れ値なし':>12}{'外れ値あり':>12}{'移動量':>12}")
    for k in before:
        print(f"{k:<8}{before[k]:>12.1f}{after[k]:>12.1f}{after[k] - before[k]:>12.1f}")

    print(f"\n  1点あたりの移動量  平均 {after['平均'] - before['平均']:.1f} 万円"
          f" vs 中央値 {after['中央値'] - before['中央値']:.1f} 万円")
    print(f"  平均の移動は外れ値の {(after['平均'] - before['平均']) / OUTLIER:.4f} 倍"
          f" ≒ 1/{x_out.size}（足した値を人数で割っただけ）")
    print(f"  標準偏差も同じ壊れ方をする  {describe.sd(x):.1f} → {describe.sd(x_out):.1f} 万円")
    print(f"  一方 IQR は  {describe.iqr(x):.1f} → {describe.iqr(x_out):.1f} 万円")

    # 外れ値の大きさを 0 から 50,000 まで動かして、2つの統計量の軌跡を見る。
    grid = np.linspace(0.0, OUTLIER, 200)
    means = np.array([describe.mean(np.append(x, v)) for v in grid])
    medians = np.array([describe.median(np.append(x, v)) for v in grid])

    fig, axes = plots.figure(1, 2, w=1.6)
    ax = axes[0]
    ax.hist(x, bins=30, color=plots.PALETTE["data"], alpha=0.55, lw=0)
    ax.axvline(before["平均"], color=plots.PALETTE["estimate"], lw=1.2)
    ax.axvline(before["中央値"], color=plots.PALETTE["alt"], lw=1.2, ls="--", dashes=(4, 2.0))
    ax.annotate(f"平均 {before['平均']:.0f}", xy=(before["平均"], 0.95),
                xycoords=("data", "axes fraction"), fontsize=6.0,
                color=plots.PALETTE["estimate"], ha="left", va="top")
    ax.annotate(f"中央値 {before['中央値']:.0f}", xy=(before["中央値"], 0.80),
                xycoords=("data", "axes fraction"), fontsize=6.0,
                color=plots.PALETTE["alt"], ha="right", va="top")
    ax.set_title("外れ値なし（n=200）")
    ax.set_xlabel("年収（万円）")
    ax.set_ylabel("人数")

    ax = axes[1]
    ax.plot(grid, means, color=plots.PALETTE["estimate"], lw=1.3)
    ax.plot(grid, medians, color=plots.PALETTE["alt"], lw=1.3, ls="--", dashes=(4, 2.0))
    ax.annotate("平均", xy=(grid[-1], means[-1]), fontsize=6.0,
                color=plots.PALETTE["estimate"], ha="right", va="bottom")
    ax.annotate("中央値", xy=(grid[-1], medians[-1]), fontsize=6.0,
                color=plots.PALETTE["alt"], ha="right", va="bottom")
    ax.set_title("足す1点を大きくしていくと")
    ax.set_xlabel("足した1点の年収（万円）")
    ax.set_ylabel("統計量（万円）")

    plots.save(fig, "fig-1-2-mean-breaks.png")


if __name__ == "__main__":
    main()
