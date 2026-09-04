"""軸を切ると差は何倍にも見える。エラーバーは長さが3通りある。

CVR 3.00% と 3.08% は 1.03 倍しか違わない。ところが軸の下端を 2.95% に置けば
棒の高さは 2.6 倍、2.99% に置けば 9 倍に見える。同じ絵にエラーバーを足すときも、
SD・SE・95%CI で長さが 70 倍違う。どれも「ばらつき」と呼ばれるが、答えている問いが
別である。図に何を描いたかは図を見ただけでは分からない。だから軸の起点と、
エラーバーの中身を必ず書く。

    uv run python examples/ch02/truncated_axis_and_errorbar_ambiguity.py
"""

import numpy as np
from scipy import stats

from toukei_tashikame import datasets, plots, testing

N, SEED = 5_000, 6
P_A, P_B = 0.0300, 0.0308
PCT = np.array([100 * P_A, 100 * P_B])
TRUNC = (2.95, 3.15)      # 軸を切る範囲（%）


def main() -> None:
    plots.setup()
    print("--- 2-8 切った軸と、3通りのエラーバー ---")
    print(f"報告された2群の CVR   A {PCT[0]:.2f}% / B {PCT[1]:.2f}%"
          f"   比 {P_B / P_A:.4f} 倍 / 差 {100 * (P_B - P_A):.2f} ポイント")

    print("\n[1] 軸の下端を変えるだけで、棒の高さの比が動く")
    print(f"{'軸の下端':<18}{'A の高さ':>10}{'B の高さ':>10}{'見かけの比':>12}")
    for bottom, label in ((0.0, "0（切らない）"), (2.90, "2.90"),
                          (TRUNC[0], f"{TRUNC[0]}"), (2.99, "2.99"), (2.999, "2.999")):
        ha, hb = PCT - bottom
        print(f"{label:<18}{ha:>10.4f}{hb:>10.4f}{hb / ha:>11.2f}倍")
    print(f"  ← データは1つで、実際の比は {P_B / P_A:.3f} 倍。"
          "変わっているのは軸の下端だけである")
    print("  下端を A の値に近づけるほど比は発散する。"
          "「何倍に見せたいか」から軸を逆算できてしまう")

    print("\n[2] エラーバーの3通りの意味（ポイント単位）")
    d = datasets.ab_test(n_a=N, n_b=N, p_a=P_A, lift=P_B / P_A - 1.0, seed=SEED)
    a, b = d.a, d.b
    sd = 100 * a.std(ddof=1)
    se = sd / np.sqrt(N)
    half = 1.96 * se
    print(f"  各 n={N:,} の 0/1 データから測る（A群、観測 CVR {100 * a.mean():.4f}%）")
    print(f"  SD     ±{sd:>7.4f} pt   1人ぶんのばらつき。0か1しか取らないので必ず大きい")
    print(f"  SE     ±{se:>7.4f} pt   CVR という推定値の精度。SD/√n = "
          f"{sd:.4f}/{np.sqrt(N):.2f}")
    print(f"  95%CI  ±{half:>7.4f} pt   1.96 × SE")
    print(f"  長さの比 SD : SE : 95%CI = {sd / half:.2f} : {se / half:.2f} : 1.00")
    print(f"  ← SD は 95%CI の {sd / half:.0f} 倍。同じ図に「エラーバー」と書いてあっても、"
          "\n    どれを描いたかで見え方が完全に変わる。SD のバーは n を増やしても縮まない")

    print("\n[3] その2群を実際に引いてみる（seed=6）")
    pa, pb = a.mean(), b.mean()
    print(f"  真値      A {100 * d.p_a:.4f}% / B {100 * d.p_b:.4f}%   （B が上）")
    print(f"  観測      A {100 * pa:.4f}%（{int(a.sum())}件）/ "
          f"B {100 * pb:.4f}%（{int(b.sum())}件）   （A が上・符号が逆転）")
    res = testing.prop_2samp(int(a.sum()), N, int(b.sum()), N)
    print(f"  2標本比率の検定  z={res.stat:.4f}, p={res.pvalue:.4f}")
    lo_a, hi_a = stats.norm.interval(0.95, loc=pa, scale=a.std(ddof=1) / np.sqrt(N))
    lo_b, hi_b = stats.norm.interval(0.95, loc=pb, scale=b.std(ddof=1) / np.sqrt(N))
    print(f"  A の 95%CI [{100 * lo_a:.4f}%, {100 * hi_a:.4f}%]")
    print(f"  B の 95%CI [{100 * lo_b:.4f}%, {100 * hi_b:.4f}%]"
          f"   重なりの幅 {100 * (min(hi_a, hi_b) - max(lo_a, lo_b)):.4f} pt")
    print(f"  ← 真の差 {100 * (P_B - P_A):.2f} pt に対して 95%CI の半幅は {half:.4f} pt。"
          "\n    区間のほうが差より広いので、n=5,000 ではそもそも見分けられない。"
          "\n    軸を切った棒グラフは、この見分けられなさを絵から消してしまう")

    # 図1: 同じ数字を、軸の下端だけ変えて2通りに描く。
    fig, axes = plots.figure(1, 2, h=1.1, w=1.8)
    for ax, bottom, top, title in (
        (axes[0], TRUNC[0], TRUNC[1], f"軸を {TRUNC[0]}–{TRUNC[1]}% に切る"),
        (axes[1], 0.0, 4.0, "軸を 0 から取る"),
    ):
        ax.bar(["A", "B"], PCT - bottom, bottom=bottom, width=0.55,
               color=[plots.PALETTE["data"], plots.PALETTE["estimate"]], lw=0, zorder=2)
        ha, hb = PCT - bottom
        ax.set_ylim(bottom, top)
        ax.set_title(f"{title}   高さの比 {hb / ha:.2f} 倍")
        ax.set_ylabel("CVR (%)")
        for i, v in enumerate(PCT):
            ax.annotate(f"{v:.2f}%", xy=(i, v), xytext=(0, 2),
                        textcoords="offset points", ha="center", fontsize=6.0)
    plots.save(fig, "fig-2-8-truncated-axis.png")

    # 図2: 同じ2本の棒に、3通りのエラーバーを付ける。縦軸も揃える。
    fig, axes = plots.figure(1, 3, h=1.05, w=2.0, sharey=True)
    obs = np.array([100 * pa, 100 * pb])
    for ax, (label, err, note) in zip(axes, [
        ("SD", sd, "1人ぶんのばらつき"),
        ("SE", se, "平均の推定精度"),
        ("95%CI", half, "1.96 × SE"),
    ], strict=True):
        ax.bar(["A", "B"], obs, width=0.55,
               color=[plots.PALETTE["data"], plots.PALETTE["estimate"]], lw=0, zorder=2)
        ax.errorbar(["A", "B"], obs, yerr=err, fmt="none", ecolor=plots.PALETTE["ink"],
                    elinewidth=0.9, capsize=3, zorder=4)
        ax.set_title(f"{label} = ±{err:.3f} pt（{note}）")
        ax.set_xlabel("群")
    axes[0].set_ylim(0, float(obs.max() + sd) * 1.15)
    axes[0].set_ylabel("観測 CVR (%)")
    plots.save(fig, "fig-2-8-errorbar-three-meanings.png")


if __name__ == "__main__":
    main()
