"""KDE の滑らかさはバンド幅が全部決めている。峰の数を数えて確かめる。

カーネル密度推定は「ヒストグラムの滑らかな版」として使われるが、滑らかさの度合いは
こちらが選ぶ。狭くすれば標本の1点1点が小さな山になり、広くすれば本物の2つの山が
1つに融ける。真の峰が −2.0 と +2.0 にあると知っている合成データで、選び方によって
何個に見えるかを数える。

    uv run python examples/ch02/kde_bandwidth_sensitivity.py
"""

import numpy as np
from scipy import stats

from toukei_tashikame import datasets, plots

N, SEED, SEP = 300, 2, 4.0
TRUE_PEAKS = (-SEP / 2, SEP / 2)      # 合成データだけが知っている真の峰
BANDWIDTHS = (0.15, 0.45, 1.20, 2.00)
GRID = np.linspace(-7.0, 7.0, 2001)


def kde_at(x: np.ndarray, h: float) -> np.ndarray:
    """バンド幅 ``h``（データと同じ単位）の KDE を GRID 上で評価する。

    ``gaussian_kde`` の ``bw_method`` は標準偏差にかける倍率なので、絶対量で
    指定したいときは標準偏差で割ってから渡す。ここを混同すると、指定した覚えの
    ない滑らかさが出る。
    """
    return stats.gaussian_kde(x, bw_method=h / x.std(ddof=1))(GRID)


def peaks_of(density: np.ndarray) -> np.ndarray:
    """密度曲線の局所最大の位置。ごく小さな凹凸も1つと数える。"""
    idx = np.flatnonzero((density[1:-1] > density[:-2]) & (density[1:-1] >= density[2:]))
    return GRID[idx + 1]


def main() -> None:
    plots.setup()
    print("--- 2-3 KDE のバンド幅で峰の数が変わる ---")

    x = datasets.bimodal(N, sep=SEP, seed=SEED)
    print(f"二峰の混合正規  n={N}  seed={SEED}  真の峰 {TRUE_PEAKS[0]:+.1f} と "
          f"{TRUE_PEAKS[1]:+.1f}  （各成分は SD=1.0、混合比 1:1）")
    print(f"標本の平均 {x.mean():.4f} / SD {x.std(ddof=1):.4f}"
          "  ← 平均は「誰もいない谷」を指している")

    # scipy の既定。factor = n^(-1/5) を標準偏差にかけたものが実効バンド幅になる。
    auto = stats.gaussian_kde(x)
    scott_h = float(auto.factor * x.std(ddof=1))
    silverman_h = float(stats.gaussian_kde(x, bw_method="silverman").factor * x.std(ddof=1))
    print(f"\n自動選択のバンド幅   Scott {scott_h:.4f} / Silverman {silverman_h:.4f}"
          f"   （factor {auto.factor:.4f} × SD {x.std(ddof=1):.4f}）")

    print(f"\n{'バンド幅':<10}{'峰の数':>7}   峰の位置")
    curves = {}
    for h in BANDWIDTHS:
        d = kde_at(x, h)
        curves[h] = d
        p = peaks_of(d)
        shown = ", ".join(f"{v:+.2f}" for v in p[:8]) + (" …" if len(p) > 8 else "")
        print(f"h={h:<8.2f}{len(p):>7}   {shown}")
    auto_d = kde_at(x, scott_h)
    auto_p = peaks_of(auto_d)
    print(f"h={scott_h:<8.4f}{len(auto_p):>7}   "
          f"{', '.join(f'{v:+.2f}' for v in auto_p)}   ← Scott の自動選択")

    err = [abs(v - t) for v, t in zip(sorted(auto_p), TRUE_PEAKS, strict=False)]
    if len(auto_p) == 2:
        print(f"  自動選択の峰は真値から {err[0]:.3f} と {err[1]:.3f} ずれている"
              "（n=300 の標本ゆらぎの範囲）")
    print("  ← 狭すぎると標本の1点1点が峰になり、広すぎると2つの山が1つに融ける。"
          "\n    「山がいくつあるか」は、バンド幅を決めてからでないと答えられない")

    fig, axes = plots.figure(2, 3, h=1.7, w=2.0, sharey=True)
    panels = [*[(f"h = {h:.2f}", kde_at(x, h)) for h in BANDWIDTHS],
              (f"Scott 自動 h = {scott_h:.3f}", auto_d)]
    axes.ravel()[-1].axis("off")
    for ax, (label, d) in zip(axes.ravel(), panels, strict=False):
        ax.hist(x, bins=30, density=True, color=plots.PALETTE["data"], alpha=0.30, lw=0)
        ax.plot(GRID, d, color=plots.PALETTE["estimate"], lw=1.2, zorder=4)
        for t in TRUE_PEAKS:
            ax.axvline(t, color=plots.PALETTE["truth"], lw=1.0, zorder=5)
        ax.set_title(f"{label} / 峰 {len(peaks_of(d))}個")
        ax.set_xlim(-7, 7)
    # 下に図が無い軸だけに x ラベルを置く。上段に置くと下段の見出しとぶつかる。
    for ax in (axes[0, 2], axes[1, 0], axes[1, 1]):
        ax.set_xlabel("x")
    fig.subplots_adjust(hspace=0.45)
    for ax in axes[:, 0]:
        ax.set_ylabel("密度")
    axes[0, 0].annotate("赤 = 真の峰 ±2.0", xy=(0.02, 0.96), xycoords="axes fraction",
                     fontsize=6.0, color=plots.PALETTE["truth"], va="top")
    plots.save(fig, "fig-2-3-kde-bandwidth.png")


if __name__ == "__main__":
    main()
