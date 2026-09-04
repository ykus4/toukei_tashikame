"""n で割ると分散は小さめに出る。10,000 回引いて偏りを数える。

「n−1 で割る」は暗記事項ではなく、数え上げれば見える事実である。真の分散 σ²=25 の
正規分布から n=10 の標本を 10,000 回引き、ddof=0（n で割る）と ddof=1（n−1 で割る）の
標本分散をそれぞれ平均する。ddof=0 の平均は 25 に届かず、ddof=1 の平均は 25 に乗る。

理由は単純で、標本分散は「標本平均からの」ずれを測っているからだ。標本平均は手元の
データに最も近い位置に来るので、母平均からのずれより必ず小さくなる。その目減り分が
ちょうど (n−1)/n 倍にあたる。

    uv run python examples/ch01/variance_ddof_bias_simulation.py
"""

import numpy as np

from toukei_tashikame import plots, sim

MU, SIGMA2, N = 0.0, 25.0, 10
TRIALS = 10_000


def one_trial(rng):
    """標本を1つ引いて、ddof=0 と ddof=1 の標本分散を返す。"""
    x = rng.normal(MU, np.sqrt(SIGMA2), size=N)
    return x.var(ddof=0), x.var(ddof=1)


def main() -> None:
    plots.setup()

    v = sim.repeat(one_trial, trials=TRIALS, seed=0, progress=False)
    v0, v1 = v[:, 0], v[:, 1]

    theory0 = SIGMA2 * (N - 1) / N   # E[ddof=0 の標本分散]
    print(f"--- σ²={SIGMA2:g}, n={N}, {TRIALS:,} 回 ---")
    print(f"{'':<22}{'平均':>10}{'理論値':>10}{'偏り率':>10}")
    print(f"{'ddof=0（n で割る）':<22}{v0.mean():>10.4f}{theory0:>10.4f}"
          f"{100 * (v0.mean() / SIGMA2 - 1):>9.1f}%")
    print(f"{'ddof=1（n−1 で割る）':<22}{v1.mean():>10.4f}{SIGMA2:>10.4f}"
          f"{100 * (v1.mean() / SIGMA2 - 1):>9.1f}%")
    print(f"\n  ddof=0 の目減りは理論上ちょうど (n−1)/n = {(N - 1) / N:.4f} 倍。"
          f"実測比 {v0.mean() / v1.mean():.4f}")
    print(f"  シミュレーション誤差（ddof=1 の平均の標準誤差）は ±{v1.std(ddof=1) / np.sqrt(TRIALS):.4f}"
          "。偏り 2.5 を説明できる大きさではない")
    print(f"  ばらつき自体は大きい。ddof=1 の 1 回分の 95% 範囲は "
          f"[{np.percentile(v1, 2.5):.2f}, {np.percentile(v1, 97.5):.2f}]")
    print("  不偏なのは「平均して当たる」ことであって、1回の推定が当たることではない")

    fig, ax = plots.figure()
    bins = np.linspace(0, 80, 60)
    ax.hist(v0, bins=bins, color=plots.PALETTE["data"], alpha=0.55, lw=0)
    ax.hist(v1, bins=bins, histtype="step", color=plots.PALETTE["estimate"], lw=1.1)
    plots.mark_truth(ax, SIGMA2, f"真の分散 = {SIGMA2:g}")
    ax.axvline(v0.mean(), color=plots.PALETTE["data"], lw=1.0, ls="--", dashes=(4, 2.0))
    ax.annotate(f"ddof=0 の平均 {v0.mean():.2f}", xy=(v0.mean(), 0.62),
                xycoords=("data", "axes fraction"), fontsize=6.0,
                color=plots.PALETTE["ink2"], ha="right", va="center",
                xytext=(-3, 0), textcoords="offset points")
    ax.annotate(f"ddof=1（平均 {v1.mean():.2f}）", xy=(0.62, 0.85), xycoords="axes fraction",
                fontsize=6.0, color=plots.PALETTE["estimate"], ha="left")
    ax.set_xlabel("標本分散")
    ax.set_ylabel("回数")
    ax.set_title(f"n={N} の標本分散を {TRIALS:,} 回")

    plots.save(fig, "fig-1-3-ddof-bias.png")


if __name__ == "__main__":
    main()
