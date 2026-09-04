"""BH と Bonferroni の優劣は、真の帰無仮説がどれだけ混ざっているかで入れ替わる。

補正の選び方は手続きの性質だけでは決まらない。**族の中に本物がどれくらい混ざって
いるか**で成績が変わるからである。BH が実際に抑える FDR は $\\frac{m_0}{m}\\alpha$ で、
本物が多い（帰無が少ない）ほど目標の α より低く、保守的な側に出る。逆にほとんどが帰無の
スクリーニングでは、ちょうど α ぎりぎりまで使い切る。

検出力の側では逆のことが起きる。Bonferroni の閾値は $\\alpha/m$ で固定なので帰無の割合に
まったく反応せず、BH は本物が多いほど有利になる。両者の差が縮んで並ぶのはどのあたりか
——真の帰無仮説の割合を 50% から 99% まで動かし、2,000 回ずつ数える。

    uv run python examples/ch09/fdr_vs_fwer_by_true_null_ratio.py
"""

import numpy as np
from scipy import special

from toukei_tashikame import plots, sim, testing

M = 100                # 仮説の総数
D_TRUE = 0.5
N = 64
ALPHA = 0.05
TRIALS = 2_000
SEED = 95
NULL_FRACTIONS = (0.50, 0.70, 0.80, 0.90, 0.95, 0.99)


def make_trial(m_true: int):
    """真の効果を ``m_true`` 本だけ持つ族を1つ作り、2手続きの成績を返す関数。"""

    def one_trial(rng) -> tuple[float, ...]:
        a = rng.normal(0.0, 1.0, size=(M, N))
        b = rng.normal(0.0, 1.0, size=(M, N))
        b[:m_true] += D_TRUE
        is_true = np.zeros(M, dtype=bool)
        is_true[:m_true] = True

        s1, s2 = a.var(axis=1, ddof=1) / N, b.var(axis=1, ddof=1) / N
        t = (b.mean(axis=1) - a.mean(axis=1)) / np.sqrt(s1 + s2)
        df = (s1 + s2) ** 2 / (s1**2 / (N - 1) + s2**2 / (N - 1))
        p = 2 * special.stdtr(df, -np.abs(t))

        out: list[float] = []
        for adj in (testing.adjust_pvalues(p, "bh"),
                    testing.adjust_pvalues(p, "bonferroni")):
            rejected = adj < ALPHA
            v = int((rejected & ~is_true).sum())
            s = int((rejected & is_true).sum())
            out += [v / (v + s) if v + s else 0.0, s / m_true, float(v > 0)]
        return tuple(out)

    return one_trial


def draw(rows: list[dict]) -> None:
    frac = np.array([r["null_fraction"] for r in rows])
    fig, axes = plots.figure(1, 2, w=2.0, h=0.95)

    ax = axes[0]
    for key, color, label in (("bh_fdr", "estimate", "BH"),
                              ("bonf_fdr", "alt", "Bonferroni")):
        y = np.array([r[key] for r in rows])
        ax.plot(frac, y, color=plots.PALETTE[color], lw=1.3, marker="o", ms=2.6, zorder=4)
        ax.annotate(label, xy=(frac[1], y[1]), xytext=(2, 5), textcoords="offset points",
                    fontsize=6.0, color=plots.PALETTE[color])
    plots.mark_truth(ax, ALPHA, "目標 FDR = 0.05", axis="y")
    ax.set_xlabel("真の帰無仮説の割合")
    ax.set_ylabel("実効 FDR")
    ax.set_ylim(0, 0.075)
    ax.set_title("BH の実効 FDR は $m_0/m \\cdot \\alpha$")

    ax = axes[1]
    for key, color, label in (("bh_power", "estimate", "BH"),
                              ("bonf_power", "alt", "Bonferroni")):
        y = np.array([r[key] for r in rows])
        ax.plot(frac, y, color=plots.PALETTE[color], lw=1.3, marker="o", ms=2.6, zorder=4)
        ax.annotate(label, xy=(frac[2], y[2]), xytext=(2, 5), textcoords="offset points",
                    fontsize=6.0, color=plots.PALETTE[color])
    ax.set_xlabel("真の帰無仮説の割合")
    ax.set_ylabel("検出力")
    ax.set_ylim(0, 1.0)
    ax.set_title("Bonferroni は帰無の割合に反応しない")

    plots.save(fig, "fig-9-5-fdr-vs-null-fraction.png")


def main() -> None:
    plots.setup()
    rows: list[dict] = []
    with sim.Timer("9-5 帰無の割合を振る"):
        for i, frac in enumerate(NULL_FRACTIONS):
            m_true = max(round(M * (1 - frac)), 1)
            out = sim.repeat(make_trial(m_true), trials=TRIALS, seed=SEED + 100 * i,
                             progress=False)
            rows.append({
                "null_fraction": frac,
                "m_true": m_true,
                "bh_fdr": float(out[:, 0].mean()),
                "bh_power": float(out[:, 1].mean()),
                "bh_fwer": float(out[:, 2].mean()),
                "bonf_fdr": float(out[:, 3].mean()),
                "bonf_power": float(out[:, 4].mean()),
                "bonf_fwer": float(out[:, 5].mean()),
            })

    print(f"仮説 {M} 本、真の効果は d={D_TRUE}、n={N}/群、α={ALPHA}、"
          f"各点 {TRIALS:,} 回\n")
    print(f"{'帰無の割合':>10}{'真の本数':>9}"
          f"{'BH FDR':>9}{'BH 検出力':>11}{'BH FWER':>10}"
          f"{'Bonf FDR':>10}{'Bonf 検出力':>13}")
    for r in rows:
        print(f"{r['null_fraction']:>10.0%}{r['m_true']:>9}"
              f"{r['bh_fdr']:>9.4f}{r['bh_power']:>11.4f}{r['bh_fwer']:>10.4f}"
              f"{r['bonf_fdr']:>10.4f}{r['bonf_power']:>13.4f}")

    first, last = rows[0], rows[-1]
    print(f"\nBH の実効 FDR は 帰無 {first['null_fraction']:.0%} で {first['bh_fdr']:.4f}、"
          f"帰無 {last['null_fraction']:.0%} で {last['bh_fdr']:.4f}。")
    print(f"BH が実際に抑えるのは $m_0/m \\cdot \\alpha$ なので、本物が多いほど目標 "
          f"{ALPHA} より低く出る（{first['null_fraction']:.0%} なら "
          f"{first['null_fraction'] * ALPHA:.4f} が目安）。")
    print(f"\n検出力は BH が {first['bh_power']:.4f} → {last['bh_power']:.4f} と落ちるのに対し、")
    print(f"Bonferroni は {first['bonf_power']:.4f} → {last['bonf_power']:.4f} でほとんど動かない。")
    print("閾値が α/m で固定だからで、族に本物が何本あろうと関係しない。")
    print(f"差が縮んで並ぶのは帰無 {last['null_fraction']:.0%} 付近"
          f"（BH {last['bh_power']:.4f} 対 Bonferroni {last['bonf_power']:.4f}）。")
    print("守りたいのが FWER なら Bonferroni、候補を絞りたいなら BH——選ぶ基準はここにある。")
    draw(rows)


if __name__ == "__main__":
    main()
