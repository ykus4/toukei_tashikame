"""崩し — 正規性検定は n が大きいほど「正規でない」と言う。そして前段に置くと歪む。

Shapiro-Wilk のような正規性検定は、帰無仮説「母集団は正規」を検定する。n が増えれば
どんな微小なずれも検出できるようになるので、**実務上は無視してよいずれ**でも棄却する。
逆に n が小さいときは、明らかに非正規でも検出できない。つまり「t 検定を使ってよいか」を
知りたいときに一番役に立たない振る舞いをする。

さらに悪いことに、正規性検定の結果を見て検定を選ぶ二段階手続きは、選択そのものが
データに依存するので第一種の誤りが名目からずれる。前段の検定で後段を選ぶこと自体が
多重性であり、「検定を選ぶための検定」は無料ではない。

    uv run python examples/ch08/normality_test_breaks_at_large_n.py
"""

import numpy as np
from scipy import stats

from toukei_tashikame import plots

SEED, ALPHA = 89, 0.05
# n=5000 の Shapiro は1回が重いので、試行数を落とす（棄却率が0か1の近くなので十分）
NS_TRIALS = ((20, 5_000), (200, 5_000), (5000, 500))

DISTS = {
    "正規 N(0,1)": lambda rng, n: rng.normal(0.0, 1.0, size=n),
    "t 分布 (df=10)": lambda rng, n: rng.standard_t(10, size=n),
}
STAGE_N, STAGE_SIGMA, STAGE_TRIALS = 10, 0.6, 15_000


def shapiro_rejection(draw, n: int, trials: int, seed: int) -> float:
    """``trials`` 本の標本それぞれに Shapiro-Wilk をかけ、棄却した割合を返す。"""
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(trials):
        if stats.shapiro(draw(rng, n)).pvalue < ALPHA:
            hits += 1
    return hits / trials


def se(rate: float, trials: int) -> float:
    return float(np.sqrt(rate * (1 - rate) / trials))


def main() -> None:
    plots.setup()

    print(f"--- 1. Shapiro-Wilk の棄却率（α={ALPHA}）---")
    print(f"{'母集団':<18}" + "".join(f"{f'n={n} ({t:,}回)':>18}" for n, t in NS_TRIALS))
    curves = {}
    for i, (name, draw) in enumerate(DISTS.items()):
        rates = [shapiro_rejection(draw, n, t, SEED + 100 * i + j)
                 for j, (n, t) in enumerate(NS_TRIALS)]
        curves[name] = rates
        print(f"{name:<20}" + "".join(f"{v:>18.4f}" for v in rates))

    t_rates = curves["t 分布 (df=10)"]
    print(f"\n  t(df=10) は正規から少ししか外れていないのに、n={NS_TRIALS[0][0]} で "
          f"{t_rates[0]:.4f} しか拾えず、n={NS_TRIALS[2][0]} では {t_rates[2]:.4f} になる。")
    print("  母集団は一つも変わっていない。変わったのは n だけである。")
    print("  「n が小さいときは通し、大きいときは落とす」——t 検定が本当に困るのは"
          "小標本のときなので、要るところで役に立たない")

    print(f"\n--- 2. 二段階手続きの第一種の誤り"
          f"（対数正規 σ={STAGE_SIGMA}、n={STAGE_N}:{STAGE_N * 3} の不均衡）---")
    rng = np.random.default_rng(SEED + 900)
    a = rng.lognormal(0.0, STAGE_SIGMA, size=(STAGE_TRIALS, STAGE_N))
    b = rng.lognormal(0.0, STAGE_SIGMA, size=(STAGE_TRIALS, STAGE_N * 3))

    p_t = stats.ttest_ind(a, b, axis=1, equal_var=False).pvalue
    p_u = stats.mannwhitneyu(a, b, axis=1).pvalue
    passed = np.array([stats.shapiro(x).pvalue >= ALPHA and stats.shapiro(y).pvalue >= ALPHA
                       for x, y in zip(a, b, strict=True)])
    p_two_stage = np.where(passed, p_t, p_u)

    rows = [
        ("常に t 検定", float((p_t < ALPHA).mean()), STAGE_TRIALS),
        ("常に Mann-Whitney", float((p_u < ALPHA).mean()), STAGE_TRIALS),
        ("正規性検定 → t / U を選ぶ", float((p_two_stage < ALPHA).mean()), STAGE_TRIALS),
        ("正規性検定を通った標本だけの t 検定", float((p_t[passed] < ALPHA).mean()),
         int(passed.sum())),
    ]
    for label, v, m in rows:
        print(f"  {label:<36} {v:.4f} ± {1.96 * se(v, m):.4f}  （{m:,}回）")

    print(f"\n  前段の正規性検定を両群とも通ったのは {passed.mean():.4f}。"
          "残りは Mann-Whitney に回っている")
    print(f"  肝心なのは最後の行で、ふるいを通った標本だけを t 検定にかけると "
          f"{rows[3][1]:.4f}。ふるいにかける前の {rows[0][1]:.4f} より**悪くなっている**")
    print("  ふるいを通るかどうかが、後段に渡す標本の形と相関している。"
          "データに依存して標本を選べば、後段の帰無分布はもう t 分布ではない")
    print(f"  二段階手続き全体としては {rows[2][1]:.4f}。名目の {ALPHA} に戻るわけでもなく、"
          f"常に Mann-Whitney を使う {rows[1][1]:.4f} より良くもならない")
    print("  「検定を選ぶための検定」は無料ではない。分布を確かめたいなら、"
          "検定ではなくヒストグラムと QQ プロットを見るほうがよい")

    # --- 図 ---
    fig, (ax1, ax2) = plots.figure(1, 2, w=1.6)

    ns = [n for n, _ in NS_TRIALS]
    for name, color in zip(DISTS, [plots.PALETTE["data"], plots.PALETTE["alt"]], strict=True):
        ax1.plot(ns, curves[name], marker="o", ms=3, lw=1.2, color=color, zorder=3)
        ax1.annotate(name, xy=(ns[1], curves[name][1]), xytext=(0, 9),
                     textcoords="offset points", ha="center", va="bottom", fontsize=6.0,
                     color=color)
    plots.mark_truth(ax1, ALPHA, "名目 α = 0.05", axis="y")
    ax1.set_xscale("log")
    ax1.set_xticks(ns)
    ax1.set_xticklabels([str(n) for n in ns])
    ax1.set_ylim(-0.05, 1.08)
    ax1.set_xlabel("n")
    ax1.set_ylabel("Shapiro-Wilk の棄却率")
    ax1.set_title("「正規でない」と言う確率は n で決まる")

    values = [v for _, v, _ in rows]
    bars = ax2.barh([3, 2, 1, 0], values, height=0.55,
                    color=[plots.PALETTE["data"], plots.PALETTE["estimate"],
                           plots.PALETTE["alt"], plots.PALETTE["reject"]], alpha=0.85, lw=0)
    for rect, v in zip(bars, values, strict=True):
        ax2.annotate(f"{v:.4f}", xy=(v, rect.get_y() + rect.get_height() / 2),
                     xytext=(3, 0), textcoords="offset points", va="center", fontsize=6.5)
    plots.mark_truth(ax2, ALPHA, "名目 α")
    ax2.set_yticks([3, 2, 1, 0])
    ax2.set_yticklabels(["常に t", "常に U", "二段階", "ふるいを\n通った標本の t"])
    ax2.set_xlim(0, max(values) * 1.35)
    ax2.set_xlabel("第一種の誤り")
    ax2.set_title(f"対数正規 σ={STAGE_SIGMA}, n={STAGE_N}:{STAGE_N * 3}"
                  f"（{STAGE_TRIALS:,}回）")

    fig.tight_layout()
    plots.save(fig, "fig-8-9-normality-test-vs-n.png")


if __name__ == "__main__":
    main()
