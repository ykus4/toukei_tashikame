"""崩し — t 検定の正規性の仮定を、4種類の非正規母集団で破ってみる。

「t 検定は正規分布を仮定する」は正しいが、**破ったときにどれくらい壊れるか**は分布の
形と n と、そして群のサイズが揃っているかで決まる。歪度5.4の対数正規でも、両群が
同じサイズなら第一種の誤りはほとんど動かない——歪みが左右の群で打ち消し合うからである。

崩れるのは**歪み × 不均衡**が重なったとき。同じ母集団・同じ n を 1:3 の不均衡に
するだけで、名目5%が6〜7%に持ち上がる。「正規性を気にしろ」ではなく「歪んだデータで
群のサイズを揃えろ」が実務的な教訓になる。

    uv run python examples/ch08/ttest_robustness_to_nonnormal.py
"""

import numpy as np
from scipy import stats

from toukei_tashikame import plots

TRIALS, SEED, ALPHA = 10_000, 85, 0.05
NS = (10, 30, 100)

# 母集団。すべて「帰無が真」— 2群は同じ母集団から引く
POPULATIONS = {
    "正規（基準）": lambda rng, shape: rng.normal(0.0, 1.0, size=shape),
    "一様": lambda rng, shape: rng.uniform(-1.0, 1.0, size=shape),
    "混合（5%が10倍の幅）": lambda rng, shape: np.where(
        rng.random(shape) < 0.05, rng.normal(0.0, 10.0, shape), rng.normal(0.0, 1.0, shape)),
    "指数": lambda rng, shape: rng.exponential(1.0, size=shape),
    "対数正規": lambda rng, shape: rng.lognormal(0.0, 1.0, size=shape),
}


def se(rate: float) -> float:
    return float(np.sqrt(rate * (1 - rate) / TRIALS))


def type1_table(ratio: int) -> tuple[dict[str, list[float]], dict[str, float]]:
    """第2群を ``ratio`` 倍の大きさにして、母集団 × n の第一種の誤りを埋める。"""
    table, skews = {}, {}
    for i, (name, draw) in enumerate(POPULATIONS.items()):
        skews[name] = float(stats.skew(draw(np.random.default_rng(SEED + 100 * i), 200_000)))
        rates = []
        for j, n in enumerate(NS):
            rng = np.random.default_rng(SEED + 100 * i + 10 * j + ratio)
            a = draw(rng, (TRIALS, n))
            b = draw(rng, (TRIALS, n * ratio))
            p = stats.ttest_ind(a, b, axis=1, equal_var=False).pvalue
            rates.append(float((p < ALPHA).mean()))
        table[name] = rates
    return table, skews


def show(title: str, table: dict[str, list[float]], skews: dict[str, float]) -> None:
    print(f"\n--- {title} ---")
    header = "".join(f"{f'n={n}':>12}" for n in NS)
    print(f"{'母集団':<24}{'歪度':>8}{header}")
    for name, rates in table.items():
        cells = "".join(f"{v:>12.4f}" for v in rates)
        print(f"{name:<26}{skews[name]:>7.2f}{cells}")


def main() -> None:
    plots.setup()
    print(f"帰無が真（2群が同じ母集団）のデータを {TRIALS:,} 組。Welch の t 検定、α={ALPHA}")

    balanced, skews = type1_table(1)
    show("均衡: 両群とも n", balanced, skews)
    unbalanced, _ = type1_table(3)
    show("不均衡: 第1群 n / 第2群 3n", unbalanced, skews)

    ln_b, ln_u = balanced["対数正規"], unbalanced["対数正規"]
    print(f"\n  対数正規（歪度 {skews['対数正規']:.2f}）の n=10:"
          f" 均衡なら {ln_b[0]:.4f} ± {1.96 * se(ln_b[0]):.4f}（むしろ保守的）、"
          f" 1:3 の不均衡だと {ln_u[0]:.4f} ± {1.96 * se(ln_u[0]):.4f}")
    print(f"  n=100 まで来ると不均衡でも {ln_u[2]:.4f}。中心極限定理が歪みを薄める")
    print(f"  正規は均衡 {balanced['正規（基準）'][0]:.4f} / 不均衡 "
          f"{unbalanced['正規（基準）'][0]:.4f} でどちらも動かない")
    print("  対称な分布（一様・混合）は歪度が0なので、不均衡にしても持ち上がらない。"
          "効いているのは正規性そのものではなく**歪み×不均衡**である")

    # --- 図 ---
    fig, (ax1, ax2) = plots.figure(1, 2, w=1.6, sharey=True)
    colors = [plots.PALETTE["data"], plots.PALETTE["prior"], plots.PALETTE["estimate"],
              plots.PALETTE["alt"], plots.PALETTE["reject"]]
    for ax, tab, title in ((ax1, balanced, "均衡 n : n"), (ax2, unbalanced, "不均衡 n : 3n")):
        for (name, rates), color in zip(tab.items(), colors, strict=True):
            ax.plot(NS, rates, marker="o", ms=3, color=color, lw=1.2, zorder=3, label=name)
        plots.mark_truth(ax, ALPHA, "名目 α = 0.05", axis="y")
        ax.set_xscale("log")
        ax.set_xticks(NS)
        ax.set_xticklabels([str(n) for n in NS])
        ax.set_xlabel("第1群の n")
        ax.set_title(title)
    # 5本が近い値で走るので、ここだけは直接ラベルより凡例のほうが読める
    ax1.legend(loc="lower right", fontsize=5.6, labelspacing=0.3, handlelength=1.4)
    ax1.set_ylabel("第一種の誤り")
    fig.tight_layout()
    plots.save(fig, "fig-8-5-nonnormal-type1.png")


if __name__ == "__main__":
    main()
