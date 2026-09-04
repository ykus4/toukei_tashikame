"""要約統計量が同じでも、絵はまるで違う — Anscombe と Datasaurus。

平均・分散・相関・回帰直線。この4つを表で見せられたら、4組のデータは「同じ」と
言うほかない。ところが散布図を描くと、直線・曲線・外れ値1点・恐竜が出てくる。
要約は捨象であって、何を捨てたかは要約からは分からない。だから描く。

    uv run python examples/ch02/anscombe_and_datasaurus_same_stats.py
"""

import numpy as np
import pandas as pd

from toukei_tashikame import datasets, plots

COLS = ["x̄", "ȳ", "sx", "sy", "r", "切片", "傾き"]


def summarize(x, y) -> dict[str, float]:
    """1組ぶんの要約。平均・分散・相関と、最小二乗の切片・傾き。"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    return {
        "x̄": x.mean(),
        "ȳ": y.mean(),
        "sx": x.std(ddof=1),
        "sy": y.std(ddof=1),
        "r": float(np.corrcoef(x, y)[0, 1]),
        "切片": float(intercept),
        "傾き": float(slope),
    }


def table(df: pd.DataFrame) -> pd.DataFrame:
    """``dataset`` 列で分けて、組ごとの要約を1行ずつ積む。"""
    rows = {name: summarize(g["x"], g["y"]) for name, g in df.groupby("dataset", sort=False)}
    return pd.DataFrame(rows).T[COLS]


def agreement(tab: pd.DataFrame, digits: int = 2) -> None:
    """丸めてから数える。「一致する」は、どの桁で見るかを決めて初めて言える。"""
    rounded = tab.round(digits)
    for c in COLS:
        vals = sorted(rounded[c].unique())
        mark = "一致" if len(vals) == 1 else f"{len(vals)}種類"
        span = tab[c].max() - tab[c].min()
        print(f"    {c:<4} 第{digits}位で {mark:<6} 生の値の幅 {span:.4f}   {vals}")


def anscombe_part() -> None:
    df = datasets.anscombe()
    tab = table(df)
    print("--- Anscombe の4組（1973）、各 n=11 ---")
    print(tab.to_string(float_format=lambda v: f"{v:.4f}"))

    print("\n  小数第2位で丸めたときの一致:")
    agreement(tab)
    print("  ← 7列すべてが1種類。回帰直線の傾きまで含めて、表の上では区別がつかない")
    print(f"  4組に共通する要約: x̄={tab['x̄'].iloc[0]:.2f} / ȳ={tab['ȳ'].iloc[0]:.2f} / "
          f"r={tab['r'].iloc[0]:.3f} / ŷ={tab['切片'].iloc[0]:.2f}+"
          f"{tab['傾き'].iloc[0]:.3f}x")

    fig, axes = plots.figure(2, 2, h=1.5, sharex=True, sharey=True)
    xs = np.array([3.0, 20.0])
    for ax, (name, g) in zip(axes.ravel(), df.groupby("dataset", sort=False), strict=True):
        ax.scatter(g["x"], g["y"], s=10, color=plots.PALETTE["data"], lw=0, zorder=3)
        s, b = np.polyfit(g["x"], g["y"], 1)
        ax.plot(xs, b + s * xs, color=plots.PALETTE["estimate"], lw=1.0, zorder=2)
        ax.set_title(f"組 {name}   ŷ = {b:.2f} + {s:.2f}x")
    fig.supxlabel("x", fontsize=7)
    fig.supylabel("y", fontsize=7)
    plots.save(fig, "fig-2-1-anscombe-quartet.png")


def datasaurus_part() -> None:
    df = datasets.datasaurus("all")
    tab = table(df)
    print("\n--- Datasaurus Dozen（13組）、各 n=142 ---")
    print(tab.to_string(float_format=lambda v: f"{v:.4f}"))

    print("\n  小数第2位で丸めたときの一致:")
    agreement(tab)
    print(f"\n  r は {tab['r'].mean():.4f} ± {tab['r'].std(ddof=1):.4f}"
          f"（最小 {tab['r'].min():.4f}、最大 {tab['r'].max():.4f}）")
    print("  ← 生の値の幅は x̄ も ȳ も 0.01 未満。丸めが2種類に割れるのは"
          "「54.265 のどちら側か」という境界の問題にすぎない")
    print("  それでも絵は13通りある")

    names = list(tab.index)
    fig, axes = plots.figure(3, 5, h=2.0, w=2.0, sharex=True, sharey=True)
    for ax, name in zip(axes.ravel(), names, strict=False):
        g = df[df["dataset"] == name]
        ax.scatter(g["x"], g["y"], s=1.2, color=plots.PALETTE["data"], lw=0, zorder=3)
        ax.set_title(name)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes.ravel()[len(names):]:
        ax.axis("off")
    plots.save(fig, "fig-2-1-datasaurus-dozen.png")


def main() -> None:
    plots.setup()
    print("--- 2-1 同じ要約、違う絵 ---")
    anscombe_part()
    datasaurus_part()


if __name__ == "__main__":
    main()
