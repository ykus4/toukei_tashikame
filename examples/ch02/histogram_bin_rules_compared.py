"""ヒストグラムの山の数は、ビン幅の選び方で変わる。

同じ342羽のヒレ長に3つのビン幅規則を当て、さらにビン数を手で振って峰を数える。
粗く切れば1つの山に見え、細かく切れば2つに割れ、もっと細かくすると度数のゆらぎが
峰に化ける。「二峰性がある」はデータの性質であると同時に、こちらが選んだ解像度の
性質でもある。だから本書は、ビン幅を1つだけ選んで結論を書かない。

    uv run python examples/ch02/histogram_bin_rules_compared.py
"""

import numpy as np

from toukei_tashikame import datasets, plots

RULES = ["sturges", "scott", "fd"]
LABEL = {"sturges": "Sturges", "scott": "Scott", "fd": "FD"}


def peak_positions(counts: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """度数列の局所最大の位置。両端はゼロで埋めて、端の山も1つと数える。"""
    padded = np.concatenate([[0], counts, [0]])
    idx = np.flatnonzero((padded[1:-1] > padded[:-2]) & (padded[1:-1] >= padded[2:]))
    return centers[idx]


def rule_width(x: np.ndarray, rule: str) -> float:
    """規則が推すビン幅そのもの。numpy が整数ビン数に丸める前の値。"""
    n = x.size
    if rule == "sturges":
        return float(np.ptp(x) / (np.ceil(np.log2(n)) + 1))
    if rule == "scott":
        return float((24.0 * np.sqrt(np.pi)) ** (1 / 3) * x.std(ddof=1) * n ** (-1 / 3))
    iqr = float(np.percentile(x, 75) - np.percentile(x, 25))
    return 2.0 * iqr * n ** (-1 / 3)


def report(x: np.ndarray, edges: np.ndarray, head: str) -> int:
    counts, _ = np.histogram(x, bins=edges)
    centers = (edges[:-1] + edges[1:]) / 2
    peaks = peak_positions(counts, centers)
    pos = ", ".join(f"{c:.0f}" for c in peaks)
    print(f"{head}{len(counts):>6}{edges[1] - edges[0]:>9.2f}mm{len(peaks):>7}   {pos}")
    return len(peaks)


def main() -> None:
    plots.setup()
    print("--- 2-2 ビン幅規則で山の数が変わる ---")

    df = datasets.penguins()
    x = df["flipper_length_mm"].to_numpy(dtype=float)
    print(f"Palmer Penguins のヒレ長 flipper_length_mm  n={x.size}"
          f"  範囲 {x.min():.0f}–{x.max():.0f}mm  s={x.std(ddof=1):.2f}mm"
          f"  IQR={np.percentile(x, 75) - np.percentile(x, 25):.1f}mm")

    print("\n[1] 3つの規則")
    print(f"{'規則':<8}{'推す幅':>9}{'ビン数':>7}{'実際の幅':>10}{'峰':>6}   峰の位置(mm)")
    chosen = {}
    for rule in RULES:
        edges = np.histogram_bin_edges(x, bins=rule)
        chosen[rule] = edges
        head = f"{LABEL[rule]:<8}{rule_width(x, rule):>8.2f}mm"
        report(x, edges, head)
    print("  ← Scott と FD は推す幅が違うのに、レンジ59mm を割ると同じ9ビンに落ちる。"
          "\n    numpy はビン数を整数に丸めるので、規則の差が消えることがある")

    print("\n[2] ビン数を手で振る")
    print(f"{'指定':<8}{'':>9}{'ビン数':>7}{'実際の幅':>10}{'峰':>6}   峰の位置(mm)")
    for k in (4, 6, 9, 15, 25, 40, 80):
        edges = np.histogram_bin_edges(x, bins=k)
        report(x, edges, f"{f'{k}ビン':<8}{'':>9}")
    print("  ← 4ビンでは山は1つ。9〜15ビンで2つに割れる。40ビンを超えると"
          "\n    度数のゆらぎが峰に化けて、数だけ増える")

    print("\n[3] 2つ目の山の正体")
    for name, g in df.groupby("species", observed=True):
        v = g["flipper_length_mm"]
        print(f"    {name:<10} n={len(v):>3}  平均 {v.mean():.1f}mm  SD {v.std(ddof=1):.1f}mm")
    print("  ← Gentoo だけが 20mm 以上長い。割れた2つ目の山はこの種である")

    fig, axes = plots.figure(2, 3, h=1.7, w=2.0, sharex=True, sharey=True)
    panels = [(LABEL[r], chosen[r]) for r in RULES] + [
        (f"手で{k}ビン", np.histogram_bin_edges(x, bins=k)) for k in (4, 25, 80)
    ]
    for ax, (label, edges) in zip(axes.ravel(), panels, strict=True):
        counts, _ = np.histogram(x, bins=edges)
        centers = (edges[:-1] + edges[1:]) / 2
        ax.hist(x, bins=edges, color=plots.PALETTE["data"], alpha=0.65, lw=0.3,
                edgecolor="white")
        for c in peak_positions(counts, centers):
            ax.axvline(c, color=plots.PALETTE["reject"], lw=0.8, ls="--", dashes=(3, 2))
        ax.set_title(f"{label}: {len(counts)}ビン / 幅{edges[1] - edges[0]:.1f}mm / "
                     f"峰{len(peak_positions(counts, centers))}")
    for ax in axes[1]:
        ax.set_xlabel("ヒレ長 (mm)")
    for ax in axes[:, 0]:
        ax.set_ylabel("度数")
    plots.save(fig, "fig-2-2-bin-rules.png")


if __name__ == "__main__":
    main()
