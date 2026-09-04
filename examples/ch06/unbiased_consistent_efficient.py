"""推定量の良さの3条件 — 不偏性・一致性・有効性を別々に測る。

同じ μ を推すのに、標本平均・標本中央値・(min+max)/2 の3つを並べる。3つとも
「平均的には当たる」（不偏）が、当たり方のばらつきは違う（有効性）。n を増やしたときの
縮み方も違う（一致性）。良い推定量とは何か、を1つの言葉で言えない理由がここにある。

    uv run python examples/ch06/unbiased_consistent_efficient.py
"""

import numpy as np

from toukei_tashikame import plots, sim

MU, SIGMA = 50.0, 10.0
N, TRIALS, SEED = 20, 10_000, 24
NAMES = ("標本平均", "標本中央値", "(min+max)/2")
SWEEP_N = (20, 50, 100, 500)


def three_estimates(rng, n: int = N) -> tuple[float, float, float]:
    """1つの標本から3つの推定値を作る。母集団も標本も、3者で全く同じもの。"""
    x = rng.normal(MU, SIGMA, size=n)
    return float(x.mean()), float(np.median(x)), float((x.min() + x.max()) / 2)


def main() -> None:
    plots.setup()
    est = sim.repeat(three_estimates, trials=TRIALS, seed=SEED, progress=False)
    bias = est.mean(axis=0) - MU
    sd = est.std(axis=0, ddof=1)

    print(f"--- 条件1・3: n={N} で {TRIALS:,} 回（真値 μ={MU:g}）---")
    print(f"  {'推定量':<12}{'バイアス':>10}{'SD':>10}{'相対効率':>10}")
    for name, b, s in zip(NAMES, bias, sd, strict=True):
        print(f"  {name:<12}{b:>+10.4f}{s:>10.4f}{(sd[0] / s) ** 2:>10.3f}")
    print(f"  バイアスはどれも 0 の付近（不偏）。10,000回のシミュレーション誤差は "
          f"±{1.96 * sd[0] / np.sqrt(TRIALS):.4f} 程度")
    print(f"  相対効率 = 標本平均の分散 / その推定量の分散。中央値は {(sd[0] / sd[1]) ** 2:.3f}"
          f"（理論 2/π = {2 / np.pi:.3f}）")
    print(f"  中央値で標本平均と同じ精度を出すには n を {1 / (sd[0] / sd[1]) ** 2:.2f} 倍 要る")

    print("\n--- 条件2: n を増やすと縮むか（一致性）---")
    curves = np.zeros((len(SWEEP_N), 3))
    for i, n in enumerate(SWEEP_N):
        e = sim.repeat(lambda rng, n=n: three_estimates(rng, n),
                       trials=TRIALS, seed=SEED + 100 * i, progress=False)
        curves[i] = e.std(axis=0, ddof=1)
    print(f"  {'n':>6}" + "".join(f"{name:>14}" for name in NAMES))
    for n, row in zip(SWEEP_N, curves, strict=True):
        print(f"  {n:>6}" + "".join(f"{v:>14.4f}" for v in row))
    ratio = curves[0] / curves[-1]
    print(f"  n を {SWEEP_N[0]} → {SWEEP_N[-1]}（{SWEEP_N[-1] // SWEEP_N[0]} 倍）にしたときのSDの縮小: "
          + " / ".join(f"{r:.2f} 倍" for r in ratio))
    print(f"  √n の法則なら {np.sqrt(SWEEP_N[-1] / SWEEP_N[0]):.2f} 倍。前の2つはこれに乗るが、"
          f"(min+max)/2 は {ratio[2]:.2f} 倍しか縮まない")
    print("  一致はするが速さが違う。「n を増やせば当たる」だけでは推定量を選べない")

    fig, axes = plots.figure(1, 2, w=2.0)
    ax = axes[0]
    bins = np.linspace(MU - 12, MU + 12, 60)
    for j, (name, style) in enumerate(zip(NAMES, ("-", "--", ":"), strict=True)):
        hist, edges = np.histogram(est[:, j], bins=bins, density=True)
        ax.plot(0.5 * (edges[1:] + edges[:-1]), hist, style,
                color=plots.PALETTE["estimate"], lw=1.1, alpha=1.0 - 0.15 * j)
        ax.annotate(f"{name}（SD {sd[j]:.2f}）", xy=(MU + 4.2, hist.max() * (0.95 - 0.22 * j)),
                    fontsize=6.0, color=plots.PALETTE["estimate"])
    plots.mark_truth(ax, MU, f"真値 {MU:g}")
    ax.set_xlabel(f"推定値（n={N}）")
    ax.set_ylabel("密度")
    ax.set_title("有効性 — 同じ中心、違う幅")

    ax = axes[1]
    for j, (name, marker) in enumerate(zip(NAMES, ("o", "s", "^"), strict=True)):
        ax.plot(SWEEP_N, curves[:, j], marker=marker, ms=3.0,
                color=plots.PALETTE["estimate"], alpha=1.0 - 0.2 * j, lw=1.0)
        ax.annotate(name, xy=(SWEEP_N[-1], curves[-1, j]), xytext=(-3, 4),
                    textcoords="offset points", ha="right", fontsize=6.0,
                    color=plots.PALETTE["estimate"])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("標本サイズ n")
    ax.set_ylabel("推定値のSD")
    ax.set_title("一致性 — n とともに 0 へ")
    plots.save(fig, "fig-6-2-three-conditions.png")


if __name__ == "__main__":
    main()
