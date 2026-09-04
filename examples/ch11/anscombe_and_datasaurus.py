"""同じ r、違う絵 — Anscombe 4組と Datasaurus 13組を相関の側から見る。

第2章では要約統計量の限界として見た2つのデータを、ここでは相関係数の話として読み直す。
Anscombe の4組は平均も分散も相関も回帰直線も一致する。r=0.816 と p 値だけを表にして
提出すれば、4組は「同じ強さの直線関係」に見える。実際には直線・曲線・外れ値1点・
1点だけ x が違う縦並び、の4つである。

Datasaurus の13組はさらに極端で、恐竜も星も同心円も、すべて r=−0.06 付近に落ちる。
相関係数が同じであることは、散布図が似ていることを何も意味しない。Pearson だけでなく
Spearman を並べても、順位相関も同じように潰れる。r を報告するときは絵も一緒に出す。

    uv run python examples/ch11/anscombe_and_datasaurus.py
"""

import numpy as np
import pandas as pd
from scipy import stats

from toukei_tashikame import datasets, plots, testing


def correlation_row(x, y) -> dict[str, float]:
    """1組ぶんの相関まわり。Pearson・Spearman・回帰直線・z 区間。"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = x.size
    r = float(np.corrcoef(x, y)[0, 1])
    lo, hi = testing.fisher_z_ci(r, n)
    slope, intercept = np.polyfit(x, y, 1)
    return {
        "n": n,
        "x̄": x.mean(),
        "ȳ": y.mean(),
        "r": r,
        "r²": r**2,
        "ρ(Spearman)": float(stats.spearmanr(x, y).statistic),
        "p": float(testing.pearson_test(x, y).pvalue),
        "CI下": lo,
        "CI上": hi,
        "切片": float(intercept),
        "傾き": float(slope),
    }


def table(df: pd.DataFrame) -> pd.DataFrame:
    rows = {name: correlation_row(g["x"], g["y"]) for name, g in df.groupby("dataset", sort=False)}
    return pd.DataFrame(rows).T


def main() -> None:
    plots.setup()

    ans = datasets.anscombe()
    tab = table(ans)
    print("--- 11-5 Anscombe の4組（各 n=11）を相関で見る ---")
    print(tab.to_string(float_format=lambda v: f"{v:.4f}"))
    print(f"\n  4組に共通: x̄={tab['x̄'].iloc[0]:.2f} / ȳ={tab['ȳ'].iloc[0]:.2f} / "
          f"r={tab['r'].iloc[0]:.3f} / r²={tab['r²'].iloc[0]:.3f} / "
          f"ŷ = {tab['切片'].iloc[0]:.2f} + {tab['傾き'].iloc[0]:.3f}x")
    print(f"  r の幅は {tab['r'].max() - tab['r'].min():.4f}、"
          f"p 値はどれも {tab['p'].min():.4f}〜{tab['p'].max():.4f} で「有意」")
    print(f"  一方 Spearman は {tab['ρ(Spearman)'].min():.4f}〜{tab['ρ(Spearman)'].max():.4f} と割れる。")
    print("  順位に直すと4組の違いが少しだけ表に出る。少しだけであって、十分ではない")

    print("\n  95%区間（Fisher の z）はどれも同じ幅で、n=11 のせいで下端が 0.4 付近まで下がる:")
    for name, row in tab.iterrows():
        print(f"    組 {name:<3} r={row['r']:.4f}   [{row['CI下']:.4f}, {row['CI上']:.4f}]")
    print("  ← 区間も重なる。数値だけでは4組を区別する材料がどこにもない")

    dino = datasets.datasaurus("all")
    dtab = table(dino)
    print("\n--- Datasaurus Dozen（13組、各 n=142）---")
    print(dtab[["r", "ρ(Spearman)", "p", "切片", "傾き"]].to_string(
        float_format=lambda v: f"{v: .4f}"))
    print(f"\n  r の平均 {dtab['r'].mean():.4f}、範囲 [{dtab['r'].min():.4f}, {dtab['r'].max():.4f}]"
          f"（幅 {dtab['r'].max() - dtab['r'].min():.4f}）")
    print(f"  p 値はどれも {dtab['p'].min():.2f} 以上。13組すべてが「相関なし」と判定される")
    print("  相関がないのは本当だが、そこから「関係がない」とは言えない。恐竜は恐竜の形をしている")

    # --- 図 ---
    names = list(dtab.index)
    fig, axes = plots.figure(3, 5, h=2.0, w=2.0, sharex=True, sharey=True)
    for ax, name in zip(axes.ravel(), names, strict=False):
        g = dino[dino["dataset"] == name]
        ax.scatter(g["x"], g["y"], s=1.2, color=plots.PALETTE["data"], lw=0, zorder=3)
        b, a = np.polyfit(g["x"], g["y"], 1)
        xs = np.array([g["x"].min(), g["x"].max()])
        ax.plot(xs, a + b * xs, color=plots.PALETTE["estimate"], lw=1.0, zorder=4)
        ax.set_title(f"{name}  r={dtab.loc[name, 'r']:.3f}")
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes.ravel()[len(names):]:
        ax.axis("off")
    plots.save(fig, "fig-11-5-datasaurus-grid.png")


if __name__ == "__main__":
    main()
