"""依頼1「この施策、効果ありました？」を、設計から報告文まで一本で通す。

実務の A/B テストは、解析より**前**で勝負がついている。CVR 3.0% を 3.6% にする
（相対 +20%）差を α=0.05・検出力 0.80 で拾うのに何件要るかを先に出しておけば、
出てきた p 値をどう読むべきかも先に決まる。

ここでは設計が要求した件数より少ない標本で回してしまった場合を追う。有意にならな
かったとき、「効果はなかった」と書くのか「分からなかった」と書くのかは、事前に計算
した検出力が決める。区間の幅を見れば、この標本が最初から何を言えなかったかが分かる。

    uv run python examples/ch18/case1_rct_did_it_work.py
"""

import numpy as np
from scipy import stats

from toukei_tashikame import datasets, plots, power, testing

P_A, P_B_TARGET = 0.030, 0.036      # 設計時に「拾いたい」と決めた効果
ALPHA, TARGET_POWER = 0.05, 0.80
N_A, N_B, LIFT, SEED = 4000, 4000, 0.20, 181


def cohens_h(p1: float, p2: float) -> float:
    """比率の効果量。角度変換してから引く（比率の差は端で意味が変わるため）。

    ``testing`` には平均差の d しか無いので、比率版はここで書く。
    0.2 で小、0.5 で中、0.8 で大、というのが慣習の目安。
    """
    return float(2 * np.arcsin(np.sqrt(p2)) - 2 * np.arcsin(np.sqrt(p1)))


def power_prop(n: int, p1: float, p2: float, alpha: float = ALPHA) -> float:
    """比率の差の検定の検出力（正規近似）。棄却限界は帰無のもとの SE で決まる。"""
    p_bar = (p1 + p2) / 2
    se_null = np.sqrt(2 * p_bar * (1 - p_bar) / n)
    se_alt = np.sqrt(p1 * (1 - p1) / n + p2 * (1 - p2) / n)
    z = stats.norm.ppf(1 - alpha / 2)
    d = abs(p2 - p1)
    return float(stats.norm.sf((z * se_null - d) / se_alt)
                 + stats.norm.cdf((-z * se_null - d) / se_alt))


def analyse(k_a: int, k_b: int) -> dict:
    """観測から、報告に要るものを全部作る。p 値だけでは報告にならない。"""
    p_a, p_b = k_a / N_A, k_b / N_B
    diff = p_b - p_a
    se = np.sqrt(p_a * (1 - p_a) / N_A + p_b * (1 - p_b) / N_B)
    half = stats.norm.ppf(1 - ALPHA / 2) * se
    res = testing.prop_2samp(k_b, N_B, k_a, N_A, method="score")
    return {"p_a": p_a, "p_b": p_b, "diff": diff, "lo": diff - half, "hi": diff + half,
            "pvalue": res.pvalue, "z": res.stat, "h": cohens_h(p_a, p_b)}


def report(a: dict, n_design: int, power_actual: float) -> str:
    """効果量・区間・仮定・限界の4点を必ず含む報告文。18-7 の型に合わせる。"""
    verdict = ("有意差あり" if a["pvalue"] < ALPHA
               else "有意差なし（＝効果がないことの証明ではない）")
    return "\n".join([
        "### 施策Xの効果検証（ランダム化比較）",
        "",
        f"- **効果量**: CVR が {100 * a['p_a']:.2f}% から {100 * a['p_b']:.2f}% へ "
        f"{100 * a['diff']:+.2f}pt（相対 {100 * (a['p_b'] / a['p_a'] - 1):+.1f}%）"
        f"、Cohen's h = {a['h']:.3f}",
        f"- **区間**: 差の 95%信頼区間 [{100 * a['lo']:+.2f}pt, {100 * a['hi']:+.2f}pt]"
        f"（p = {a['pvalue']:.4f}、{verdict}）",
        "- **仮定**: 割付は乱数によるランダム化（交絡は期待値の意味で消えている）／"
        "各ユーザの観測は独立／正規近似が効く程度に件数がある",
        f"- **限界**: 設計上 {n_design:,}件/群が必要だったところ {N_A:,}件/群で打ち切った。"
        f"想定した +{100 * (P_B_TARGET - P_A):.1f}pt に対する検出力は {power_actual:.2f} "
        f"しかなく、この標本は最初から「差を見つけられない」側に寄っている。"
        f"区間の幅 {100 * (a['hi'] - a['lo']):.2f}pt が、言えることの限界そのもの",
    ])


def draw(a: dict) -> None:
    fig, axes = plots.figure(1, 2, w=2.0, h=0.95)
    pal = plots.PALETTE

    # ① 2群の CVR と、それぞれの 95% 区間
    ax = axes[0]
    for i, (label, p, n) in enumerate((("A（対照）", a["p_a"], N_A),
                                       ("B（施策）", a["p_b"], N_B))):
        se = np.sqrt(p * (1 - p) / n)
        half = stats.norm.ppf(1 - ALPHA / 2) * se
        ax.plot([100 * (p - half), 100 * (p + half)], [i, i],
                color=pal["interval"], lw=2.0, solid_capstyle="butt", zorder=3)
        ax.scatter([100 * p], [i], s=18, color=pal["estimate"], zorder=4)
        ax.annotate(f"{label}  {100 * p:.2f}%", xy=(100 * p, i), xytext=(0, 7),
                    textcoords="offset points", ha="center", fontsize=6.2,
                    color=pal["ink2"])
    plots.mark_truth(ax, 100 * P_A, f"A の真値 {100 * P_A:.1f}%")
    ax.axvline(100 * P_A * (1 + LIFT), color=pal["truth"], lw=1.0, ls="--",
               dashes=(4, 2.2), zorder=5)
    ax.annotate(f"B の真値 {100 * P_A * (1 + LIFT):.1f}%",
                xy=(100 * P_A * (1 + LIFT), 0.02), xycoords=("data", "axes fraction"),
                fontsize=6.0, color=pal["truth"], ha="left", va="bottom",
                xytext=(2, 0), textcoords="offset points")
    ax.set_ylim(-0.6, 1.8)
    ax.set_yticks([])
    ax.set_xlabel("CVR（%）")
    ax.set_title("① 観測された CVR と 95%信頼区間")

    # ② 差の区間。設計時に狙った効果が、区間のどこに入っているか
    ax = axes[1]
    x = np.linspace(100 * (a["diff"] - 4 * (a["hi"] - a["diff"]) / 1.96),
                    100 * (a["diff"] + 4 * (a["hi"] - a["diff"]) / 1.96), 400)
    se_pt = 100 * (a["hi"] - a["diff"]) / stats.norm.ppf(1 - ALPHA / 2)
    ax.plot(x, stats.norm.pdf(x, 100 * a["diff"], se_pt), color=pal["estimate"],
            lw=1.2, zorder=4)
    plots.mark_interval(ax, 100 * a["lo"], 100 * a["hi"], label="95%信頼区間")
    ax.axvline(0.0, color=pal["ink2"], lw=0.8, zorder=5)
    ax.annotate("差なし", xy=(0, 0.96), xycoords=("data", "axes fraction"),
                xytext=(3, 0), textcoords="offset points", fontsize=6.0,
                color=pal["ink2"], va="top")
    plots.mark_truth(ax, 100 * (P_A * LIFT), f"真の差 {100 * P_A * LIFT:+.2f}pt")
    ax.set_xlabel("B − A の差（pt）")
    ax.set_ylabel("密度")
    ax.set_title("② 区間は 0 も真値も含む＝どちらとも言えない")

    fig.tight_layout()
    plots.save(fig, "fig-18-2-rct-effect-and-ci.png")


def main() -> None:
    plots.setup()

    # --- ① 事前設計。データを見る前に済ませておく ---
    n_design = power.n_for_proportions(P_A, P_B_TARGET, power=TARGET_POWER, alpha=ALPHA)
    print(f"--- 18-2 依頼1「この施策、効果ありました？」（seed={SEED}）---\n")
    print("① 事前設計（データを見る前にやること）")
    print(f"  拾いたい効果      CVR {100 * P_A:.1f}% → {100 * P_B_TARGET:.1f}%"
          f"（+{100 * (P_B_TARGET - P_A):.1f}pt、相対 +{100 * (P_B_TARGET / P_A - 1):.0f}%）")
    print(f"  α / 検出力        {ALPHA} / {TARGET_POWER}")
    print(f"  必要な件数        {n_design:,} 件/群（両群で {2 * n_design:,} 件）")

    # --- ② 実際に取れた標本。設計を満たしていない ---
    d = datasets.ab_test(n_a=N_A, n_b=N_B, p_a=P_A, lift=LIFT, seed=SEED)
    k_a, k_b = int(d.a.sum()), int(d.b.sum())
    power_actual = power_prop(N_A, P_A, P_B_TARGET)
    print(f"\n② 実際に取れたのは {N_A:,} 件/群（設計の {N_A / n_design:.0%}）")
    print(f"  この件数での検出力 {power_actual:.4f}   ← 拾いたかった差があっても、"
          f"{1 - power_actual:.0%} は見逃す")

    a = analyse(k_a, k_b)
    print(f"\n③ 解析（真値は A {d.p_a:.3f} / B {d.p_b:.4f}。合成データだけが知っている）")
    print(f"  A  {k_a:>4} / {N_A:,} = {100 * a['p_a']:.2f}%")
    print(f"  B  {k_b:>4} / {N_B:,} = {100 * a['p_b']:.2f}%")
    print(f"  差            {100 * a['diff']:+.2f}pt"
          f"（真の差は {100 * (d.p_b - d.p_a):+.2f}pt）")
    print(f"  95%信頼区間   [{100 * a['lo']:+.2f}pt, {100 * a['hi']:+.2f}pt]"
          f"  幅 {100 * (a['hi'] - a['lo']):.2f}pt")
    print(f"  z / p 値      {a['z']:.4f} / {a['pvalue']:.4f}")
    print(f"  Cohen's h     {a['h']:.4f}"
          f"（設計時に想定した h は {cohens_h(P_A, P_B_TARGET):.4f}）")

    print("\n④ 読み方")
    if a["pvalue"] < ALPHA:
        print("  有意。ただし報告すべきは p 値ではなく、区間の下端と上端の両方である")
    else:
        print("  有意ではない。だが区間は真の差 "
              f"{100 * (d.p_b - d.p_a):+.2f}pt も 0 も含んでおり、")
        print("  「効果がなかった」ではなく**「この標本では分からなかった」**が正しい。")
        print(f"  検出力 {power_actual:.2f} の設計で有意にならないのは、"
              "むしろ起こるほうが普通")

    print("\n⑤ 報告文（18-7 のテンプレート）\n")
    print(report(a, n_design, power_actual))
    print()
    draw(a)


if __name__ == "__main__":
    main()
