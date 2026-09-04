"""無補正・Bonferroni・Holm・BH が何を守り、何を捨てるかを表にする。

補正の選択は「どれが厳しいか」ではなく「何を守りたいか」で決まる。Bonferroni と Holm
は**族全体で1つでも誤るか**（FWER）を抑え、BH は**棄却したうちの誤りの割合**（FDR）を
抑える。抑えている量が違うので、同じ土俵で強弱を比べても意味がない。

100 仮説のうち 10 本だけに真の効果（d=0.5）がある設定を 2,000 回作り、4つの手続きの
FWER・FDR・検出力を並べる。あわせて、手書きの BH 補正を statsmodels の
``multipletests`` と突き合わせる。

    uv run python examples/ch09/bonferroni_holm_bh_tradeoff.py
"""

import numpy as np
from scipy import special
from statsmodels.stats.multitest import multipletests

from toukei_tashikame import plots, sim, testing

M, M_TRUE = 100, 10          # 仮説の総数と、そのうち真に効果があるもの
D_TRUE = 0.5                 # 真の効果量（Cohen の d）
N = 64                       # 群あたりの人数（d=0.5 で検出力80%になる大きさ）
ALPHA = 0.05
TRIALS = 2_000
SEED = 94

METHODS = ["無補正", "Bonferroni", "Holm", "BH（FDR）"]


def one_trial(rng) -> tuple[float, ...]:
    """100 本の検定を1回ぶん回し、4手続きの (FWER指標, FDP, 検出力) を返す。"""
    a = rng.normal(0.0, 1.0, size=(M, N))
    b = rng.normal(0.0, 1.0, size=(M, N))
    b[:M_TRUE] += D_TRUE                       # 先頭 10 本だけ真の効果を入れる
    is_true = np.zeros(M, dtype=bool)
    is_true[:M_TRUE] = True

    s1, s2 = a.var(axis=1, ddof=1) / N, b.var(axis=1, ddof=1) / N
    t = (b.mean(axis=1) - a.mean(axis=1)) / np.sqrt(s1 + s2)
    df = (s1 + s2) ** 2 / (s1**2 / (N - 1) + s2**2 / (N - 1))
    p = 2 * special.stdtr(df, -np.abs(t))

    adjusted = [
        p,
        testing.adjust_pvalues(p, "bonferroni"),
        testing.adjust_pvalues(p, "holm"),
        testing.adjust_pvalues(p, "bh"),
    ]

    out: list[float] = []
    for adj in adjusted:
        rejected = adj < ALPHA
        false_pos = int((rejected & ~is_true).sum())      # V
        true_pos = int((rejected & is_true).sum())        # S
        total = false_pos + true_pos                      # R
        out += [
            float(false_pos > 0),                         # 族に1つでも誤りがあるか
            false_pos / total if total else 0.0,           # 誤り発見の割合 FDP
            true_pos / M_TRUE,                            # 検出力
        ]

    # 手書きの BH と statsmodels を突き合わせる。
    lib = multipletests(p, alpha=ALPHA, method="fdr_bh")[1]
    out.append(float(np.max(np.abs(adjusted[3] - lib))))
    return tuple(out)


def draw(fwer: np.ndarray, fdr: np.ndarray, power: np.ndarray) -> None:
    fig, ax = plots.figure(h=1.05)
    x = np.arange(len(METHODS))
    width = 0.27
    bars = [
        (fwer, -width, plots.PALETTE["reject"], "FWER（1つでも誤る確率）"),
        (fdr, 0.0, plots.PALETTE["alt"], "FDR（棄却のうち誤りの割合）"),
        (power, width, plots.PALETTE["estimate"], "検出力（真の10本を拾う割合）"),
    ]
    for values, off, color, label in bars:
        ax.bar(x + off, values, width=width * 0.92, color=color, lw=0, label=label)
        for xi, v in zip(x + off, values, strict=True):
            ax.annotate(f"{v:.3f}", xy=(xi, v), xytext=(0, 2), textcoords="offset points",
                        ha="center", fontsize=5.4, color=plots.PALETTE["ink2"])
    plots.mark_truth(ax, ALPHA, "目標 0.05", axis="y")
    ax.set_xticks(x, METHODS, fontsize=6.4)
    ax.set_ylabel("割合")
    ax.set_xlim(-0.55, len(METHODS) - 0.28)   # 右端に「目標 0.05」の注記を置く余白
    ax.set_ylim(0, 1.12)
    ax.legend(loc="upper center", ncol=1, fontsize=5.8)
    ax.set_title(f"100仮説中10本が真（d={D_TRUE}, n={N}/群, {TRIALS:,}回）")
    plots.save(fig, "fig-9-4-correction-tradeoff.png")


def main() -> None:
    plots.setup()
    with sim.Timer("9-4 補正の取捨"):
        out = sim.repeat(one_trial, trials=TRIALS, seed=SEED, progress=False)

    fwer = out[:, 0::3][:, :4].mean(axis=0)
    fdr = out[:, 1::3][:, :4].mean(axis=0)
    power = out[:, 2::3][:, :4].mean(axis=0)
    max_diff = out[:, -1].max()

    print(f"仮説 {M} 本のうち {M_TRUE} 本が真（d={D_TRUE}）、n={N}/群、"
          f"α={ALPHA}、{TRIALS:,} 回\n")
    print(f"{'手続き':<14}{'FWER':>9}{'FDR':>9}{'検出力':>9}{'抑えている量':>16}")
    guards = ["（何も抑えない）", "FWER", "FWER", "FDR"]
    for name, f, q, pw, g in zip(METHODS, fwer, fdr, power, guards, strict=True):
        print(f"{name:<14}{f:>9.4f}{q:>9.4f}{pw:>9.4f}{g:>16}")

    print(f"\n無補正では {fwer[0]:.4f} の確率で偽陽性が混じる。100本も並べれば当然である。")
    print(f"Bonferroni と Holm は FWER を {fwer[1]:.4f} / {fwer[2]:.4f} に抑えるが、")
    print(f"検出力は {power[0]:.4f} → {power[1]:.4f} / {power[2]:.4f} まで落ちる。")
    print(f"BH は FWER を諦めるかわりに（{fwer[3]:.4f}）、FDR を {fdr[3]:.4f} に保ったまま")
    print(f"検出力 {power[3]:.4f} を残す。探索的に候補を絞る場面ではこちらが合う。")
    print(f"\n手書き BH と statsmodels multipletests の最大差: {max_diff:.3g}")
    draw(fwer, fdr, power)


if __name__ == "__main__":
    main()
