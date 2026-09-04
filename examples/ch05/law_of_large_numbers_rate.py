"""大数の法則 — 収束する。ただし √n の速さでしか収束しない。

指数分布（真の平均 2.0）の累積平均を n=10⁶ まで20本の軌跡で追う。どの軌跡も最後には
2.0 に吸い寄せられる。それが大数の法則だが、この本が見たいのは**縮み方の速さ**である。

|x̄ − μ| を n の関数として両対数にプロットすると傾きが −1/2 になる。誤差を10分の1に
するには n を100倍にしなければならない、というのがこの傾きの意味であり、
「データを増やせば増やすほど正確」の中身はこの一行に尽きる。

    uv run python examples/ch05/law_of_large_numbers_rate.py
"""

import unicodedata

import numpy as np

from toukei_tashikame import plots

MU = 2.0            # 指数分布の真の平均（σ も 2.0）
N_MAX = 1_000_000
N_PATHS = 20
SEED = 18
CHECKPOINTS = [10, 100, 1_000, 10_000, 100_000, 1_000_000]


def running_mean_at(rng, checkpoints, keep_idx):
    """1本ぶんの累積平均を作り、検査点の値と描画用の間引き軌跡を返す。"""
    x = rng.exponential(scale=MU, size=N_MAX)
    running = np.cumsum(x) / np.arange(1, N_MAX + 1)
    return running[np.asarray(checkpoints) - 1], running[keep_idx]


def rj(text: str, width: int) -> str:
    """全角を2桁として数えて右詰めする。日本語の見出しでも表の桁が揃う。"""
    w = sum(0 if unicodedata.combining(c) else
            2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in text)
    return " " * max(width - w, 0) + text


def main() -> None:
    plots.setup()
    keep_idx = np.unique(np.logspace(0, np.log10(N_MAX), 400).astype(int)) - 1
    rng = np.random.default_rng(SEED)
    at_check, paths = [], []
    for _ in range(N_PATHS):
        c, p = running_mean_at(rng, CHECKPOINTS, keep_idx)
        at_check.append(c)
        paths.append(p)
    at_check = np.array(at_check)        # (20, 6)
    paths = np.array(paths)              # (20, 400)

    dev = np.abs(at_check - MU)
    print(f"--- |x̄ − μ| の縮み方（指数分布, 真の平均 {MU}, {N_PATHS}本の平均）---")
    print("  " + rj("n", 9) + "  " + rj("|x̄−μ| の平均", 13) + "  " + rj("最大", 8)
          + "  " + rj("理論 σ√(2/πn)", 13))
    for j, n in enumerate(CHECKPOINTS):
        theory = MU * np.sqrt(2.0 / (np.pi * n))
        print(f"  {n:>9,}  {dev[:, j].mean():>13.4f}  {dev[:, j].max():>8.4f}"
              f"  {theory:>13.4f}")

    slope, _intercept = np.polyfit(np.log10(CHECKPOINTS), np.log10(dev.mean(axis=0)), 1)
    print(f"\n  両対数の傾き = {slope:.3f}   （理論 −0.500）")
    print(f"  ※ {N_PATHS}本の平均なので、傾きの推定自体が ±0.03 程度は動く")
    print(f"  10⁻¹ 誤差を 10⁻² にするには n を {10 ** (-1 / slope):.0f} 倍にする必要がある")
    print(f"\n  n=10⁶ でも1本目の累積平均は {at_check[0, -1]:.4f}"
          f"（μ からのずれ {at_check[0, -1] - MU:+.4f}）。0 にはならない")

    fig, (ax1, ax2) = plots.figure(1, 2, w=2.0)
    xs = keep_idx + 1
    for p in paths:
        ax1.plot(xs, p, color=plots.PALETTE["estimate"], lw=0.5, alpha=0.35)
    ax1.plot(xs, MU + 1.96 * MU / np.sqrt(xs), color=plots.PALETTE["ink2"], lw=0.8,
             ls="--", dashes=(4, 2.0))
    ax1.plot(xs, MU - 1.96 * MU / np.sqrt(xs), color=plots.PALETTE["ink2"], lw=0.8,
             ls="--", dashes=(4, 2.0))
    plots.mark_truth(ax1, MU, f"真の平均 = {MU:g}", axis="y")
    ax1.set_xscale("log")
    ax1.set_ylim(MU - 1.6, MU + 1.6)
    ax1.set_xlabel("n（対数）")
    ax1.set_ylabel("累積平均")
    ax1.set_title(f"{N_PATHS}本の累積平均と ±1.96σ/√n")

    ax2.plot(CHECKPOINTS, dev.mean(axis=0), "o-", color=plots.PALETTE["estimate"],
             ms=3, lw=1.0)
    ax2.plot(CHECKPOINTS, MU * np.sqrt(2.0 / (np.pi * np.array(CHECKPOINTS))),
             color=plots.PALETTE["truth"], lw=1.1)
    ax2.annotate("理論 σ√(2/πn)", xy=(1e3, MU * np.sqrt(2.0 / (np.pi * 1e3))),
                 xytext=(4, 5), textcoords="offset points", fontsize=6.0,
                 color=plots.PALETTE["truth"])
    ax2.annotate(f"実測の傾き {slope:.3f}", xy=(1e4, dev.mean(axis=0)[3]),
                 xytext=(4, -12), textcoords="offset points", fontsize=6.0,
                 color=plots.PALETTE["estimate"])
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("n（対数）")
    ax2.set_ylabel("|x̄ − μ|（対数）")
    ax2.set_title("両対数で傾き −1/2")
    plots.save(fig, "fig-5-4-lln-rate.png")


if __name__ == "__main__":
    main()
