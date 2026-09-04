"""「第90百分位」は1つに決まらない。numpy の9つの method を並べて確かめる。

n=11 のデータで第90百分位を求める。順位でいえば 0.9×(11−1)+1 = 第10位ちょうど……に
見えるが、その「ちょうど」の決め方が9通りある。numpy の ``percentile`` は
``method`` 引数でそれを選べるようになっていて、既定は ``linear`` である。

小標本では選び方で値が数点動く。「95パーセンタイルのレイテンシ」を2つのツールで
測って食い違うとき、たいていは実装の優劣ではなくこの分岐である。既定の ``linear`` が
何をしているかは、下の ``linear_by_hand`` の6行がすべてだ。

    uv run python examples/ch01/percentile_nine_interpolations.py
"""

import numpy as np

from toukei_tashikame import datasets, plots

Q = 0.90
METHODS = [
    "inverted_cdf",
    "averaged_inverted_cdf",
    "closest_observation",
    "interpolated_inverted_cdf",
    "hazen",
    "weibull",
    "linear",
    "median_unbiased",
    "normal_unbiased",
]


def linear_by_hand(x, q: float) -> float:
    """numpy の既定 ``linear`` を手で書く。これがやっていることの全部。"""
    a = np.sort(np.asarray(x, dtype=float))
    h = (a.size - 1) * q            # 0 始まりの「小数の順位」
    lo = int(np.floor(h))
    if lo >= a.size - 1:            # 右端は補間する相手がいない
        return float(a[-1])
    frac = h - lo
    return float(a[lo] + frac * (a[lo + 1] - a[lo]))


def main() -> None:
    plots.setup()

    x = datasets.normal_sample(n=11, mu=70.0, sigma=10.0, seed=3)
    ordered = np.sort(x)

    print(f"--- n={x.size} のデータ（昇順）---")
    print("  " + "  ".join(f"{v:.2f}" for v in ordered))
    print(f"\n--- 第{Q:.0%}百分位を9つの method で ---")
    values = []
    for m in METHODS:
        v = float(np.percentile(x, 100 * Q, method=m))
        values.append(v)
        print(f"  {m:<26}{v:>8.2f}")
    values = np.array(values)

    print(f"\n  最小 {values.min():.2f} / 最大 {values.max():.2f} / "
          f"レンジ {values.max() - values.min():.2f}"
          f"（データの範囲 {ordered.max() - ordered.min():.2f} の "
          f"{100 * (values.max() - values.min()) / (ordered.max() - ordered.min()):.1f}%）")

    hand = linear_by_hand(x, Q)
    lib = float(np.percentile(x, 100 * Q, method="linear"))
    print(f"\n  linear（numpy） {lib:.6f}")
    print(f"  linear（手書き） {hand:.6f}")
    print(f"  差               {abs(hand - lib):.6f}   ← 既定の中身は6行で書ける")

    h = (x.size - 1) * Q
    print(f"  小数の順位 h = (n−1)×q = {h:.1f}。"
          f"第{int(np.floor(h)) + 1}位に重み {1 - h % 1:.1f}、"
          f"第{int(np.floor(h)) + 2}位に重み {h % 1:.1f}")
    if h % 1 == 0:
        print(f"  今回は h がちょうど整数なので、linear は第{int(h) + 1}位のデータ点そのものを返している")

    # 補間が効く q でも一致するかを、0.01 刻みで総当たりして確かめる。
    grid = np.arange(0.01, 1.00, 0.01)
    diffs = np.array([abs(linear_by_hand(x, q) - float(np.percentile(x, 100 * q))) for q in grid])
    print(f"  q を 0.01〜0.99 で総当たりしても最大差は {diffs.max():.6f}")
    print(f"  たとえば第25百分位なら 手書き {linear_by_hand(x, 0.25):.4f} / "
          f"numpy {float(np.percentile(x, 25)):.4f}（h={(x.size - 1) * 0.25:.1f} なので今度は本当に内分する）")

    fig, ax = plots.figure(h=1.15)
    y = np.arange(len(METHODS))
    ax.scatter(values, y, s=18, color=plots.PALETTE["estimate"], zorder=4, lw=0)
    for xi in ordered[-4:]:
        ax.axvline(xi, color=plots.PALETTE["data"], lw=0.7, alpha=0.6, zorder=1)
        ax.annotate(f"{xi:.1f}", xy=(xi, 0.01), xycoords=("data", "axes fraction"),
                    fontsize=5.6, color=plots.PALETTE["ink2"], ha="center", va="bottom",
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 0.6})
    ax.set_yticks(y)
    ax.set_yticklabels(METHODS, fontsize=5.8)
    ax.set_xlabel(f"第{Q:.0%}百分位")
    ax.set_title("同じ11個のデータ、9通りの答え（縦線は上位4データ点）")

    plots.save(fig, "fig-1-4-percentile-methods.png")


if __name__ == "__main__":
    main()
