"""依頼2「ログしかないけど効果を出して」に、どこまで答えられるかを詰める。

ランダム化されていないログで因果を言うには、**未観測の交絡が無い**という検証不能な
仮定を置くしかない。置いてよいかどうかはデータからは分からないので、代わりにやること
は2つある。ひとつは調整の梯子を順に登って推定値がどれだけ動くかを見せること、もう
ひとつは「結論をひっくり返すのに必要な未観測交絡の強さ」（E値）を出すこと。

真の効果が 0 のデータで試す。素朴な差は交絡だけで +2.4 を作り出す。交絡因子が記録
されていれば 0 に戻せるが、記録されていなければ戻す手段は無い。最後は判定ロジックに
かけて、答えてよいかどうかを機械的に決める。

    uv run python examples/ch18/case2_observational_can_we_answer.py
"""

import numpy as np

from toukei_tashikame import causal, datasets, plots, regression, testing

N, ATE_TRUE, CONFOUNDING, SEED = 4000, 0.0, 2.0, 183
SMD_OK = 0.10          # 共変量が揃ったとみなす慣習的な閾値
PS_EXTREME = 0.05      # これより 0/1 に近い傾向スコアは正値性の赤信号


def d_to_rr(d: float) -> float:
    """標準化平均差をおおよそのリスク比に直す（VanderWeele の近似 RR ≈ exp(0.91 d)）。

    E値はリスク比を入口に取る道具なので、連続量の効果を報告するときはここを通す。
    ``causal.e_value`` は RR しか受け取らないため、変換はスクリプト側で持つ。
    """
    return float(np.exp(0.91 * d))


def judge(checks: dict[str, bool]) -> str:
    """答えてよいかを機械的に決める。人が「まあ大丈夫だろう」と言う余地を残さない。"""
    return "ANSWER: IDENTIFIED" if all(checks.values()) else "ANSWER: NOT IDENTIFIED"


def draw(rows: list[tuple[str, float, float, float]]) -> None:
    """調整の梯子。上に行くほど強い仮定を置いている、という順に並べる。"""
    fig, ax = plots.figure(w=1.35, h=0.8)
    pal = plots.PALETTE

    for i, (_label, est, lo, hi) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.plot([lo, hi], [y, y], color=pal["interval"], lw=2.0,
                solid_capstyle="butt", zorder=3)
        ax.scatter([est], [y], s=20, color=pal["estimate"], zorder=4)
        ax.annotate(f"{est:+.3f}", xy=(est, y), xytext=(0, 6),
                    textcoords="offset points", ha="center", fontsize=6.0,
                    color=pal["estimate"])
    plots.mark_truth(ax, ATE_TRUE, f"真の ATE = {ATE_TRUE:g}")
    ax.set_ylim(-0.6, len(rows) - 0.3)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in reversed(rows)])
    ax.set_xlabel("推定された平均処置効果")
    ax.set_title("調整の梯子。下ほど強い仮定に寄りかかっている")
    fig.tight_layout()
    plots.save(fig, "fig-18-3-estimates-ladder.png")


def main() -> None:
    plots.setup()
    d = datasets.observational(n=N, ate=ATE_TRUE, confounding=CONFOUNDING, seed=SEED)
    x, y, z = d.x, d.y, d.z

    print(f"--- 18-3 依頼2「ログしかない」（n={N:,}, 真の ATE={ATE_TRUE:g}, "
          f"seed={SEED}）---\n")
    print(f"  処置群 {int(x.sum()):,}件 / 対照群 {int((1 - x).sum()):,}件。"
          "割付は乱数ではなく z（利用意欲）に従っている")

    # --- ① 素朴な群間差。実務で最初に出てくる数字 ---
    naive = causal.naive_diff(y, x)
    print("\n① 素朴な群間差 — これを効果と呼んではいけない")
    print(f"  推定 {naive.estimate:+.4f}   95%CI "
          f"[{naive.ci[0]:+.4f}, {naive.ci[1]:+.4f}]")
    print(f"  真値 {ATE_TRUE:+.4f} に対して {naive.estimate - ATE_TRUE:+.4f} のずれ。"
          "効果はゼロなのに、交絡だけでこれだけ出る")

    bal_before = causal.balance_table(z, x, names=["z（利用意欲）"])
    print(f"  共変量バランス SMD = {bal_before['SMD'].iloc[0]:.3f}"
          f"（|SMD| < {SMD_OK} なら揃った）← 2群は最初から別物")

    # --- ② z が記録されている場合。共変量調整 ---
    X = np.column_stack([np.ones(N), x, z])
    ols = regression.ols(X, y, add_const=False, names=["const", "処置", "z"])
    lo_ols, hi_ols = ols.conf_int()[1]
    print("\n② z が記録されていた場合 — 共変量調整つき回帰")
    print(f"  推定 {ols.b[1]:+.4f}   95%CI [{lo_ols:+.4f}, {hi_ols:+.4f}]"
          f"   （SE {ols.se[1]:.4f}）")

    # --- ③ 傾向スコアと IPW ---
    ps = causal.propensity_score(np.column_stack([np.ones(N), z]), x)
    ipw = causal.ipw_ate(y, x, ps, stabilized=True)
    w = np.where(x == 1, 1.0 / ps, 1.0 / (1.0 - ps))
    bal_after = causal.balance_table(z, x, weights=w, names=["z（利用意欲）"])
    n_extreme = int(((ps < PS_EXTREME) | (ps > 1 - PS_EXTREME)).sum())
    print("\n③ 傾向スコア + IPW（同じく z が要る）")
    print(f"  推定 {ipw.estimate:+.4f}   95%CI "
          f"[{ipw.ci[0]:+.4f}, {ipw.ci[1]:+.4f}]")
    print(f"  重み付け後の SMD {bal_after['SMD'].iloc[0]:+.4f}"
          f"（前は {bal_before['SMD'].iloc[0]:+.3f}）← 揃った")
    print(f"  ただし {ps.min():.4f} 〜 {ps.max():.4f} まで散った傾向スコアのうち、"
          f"0/1 から {PS_EXTREME} 以内が {n_extreme}件")
    print(f"  重みの最大 {ipw.extra['重みの最大']}、"
          f"{ipw.extra['切り落とし']}を切り落として有効標本 {ipw.extra['有効標本']}")
    print("  ← 「その共変量では処置されない人」がいる。正値性が怪しい")

    # --- ④ z が記録されていない場合。E値で開き直る ---
    d_naive = testing.cohens_d(y[x == 1], y[x == 0])
    sp = np.sqrt((y[x == 1].var(ddof=1) + y[x == 0].var(ddof=1)) / 2)
    rr = d_to_rr(d_naive)
    rr_lo = d_to_rr(naive.ci[0] / sp)
    ev = causal.e_value(rr)
    ev_lo = causal.e_value(rr, rr_lo)
    # 既知の交絡 z が、同じ尺度でどれだけ強いかを測る。比較の相手がないと E値は読めない。
    d_z = testing.cohens_d(z[x == 1], z[x == 0])
    rr_z = d_to_rr(d_z)
    print("\n④ z が記録されていなかった場合 — E値（未観測交絡への感度）")
    print(f"  素朴差 {naive.estimate:+.4f} を標準化すると d = {d_naive:.4f}"
          f" → RR 換算 {rr:.2f}")
    print(f"  E値              {ev:.2f}   ← この関連を消すには、"
          f"未観測交絡が処置とも結果とも RR {ev:.2f} で結びついている必要がある")
    print(f"  95%CI 下限の E値 {ev_lo:.2f}   ← 「効果あり」を消すのに要る強さ")
    print(f"  参考: 既知の交絡 z の強さは RR 換算 {rr_z:.2f}。"
          "この領域には現に E値級の交絡が存在する")
    print("  ← E値が大きくても安心材料にならない。同じ強さの変数が実在するなら、"
          "他にもあると考えるべき")

    # --- ⑤ 判定 ---
    checks = {
        "処置がランダムに割り付けられている": False,
        "交絡因子がすべて記録されている（未観測交絡なし）": False,
        f"正値性: 極端な傾向スコアが無い（{PS_EXTREME} 以内が0件）": n_extreme == 0,
        f"重み付け後の共変量バランス |SMD| < {SMD_OK}":
            bool(abs(bal_after["SMD"].iloc[0]) < SMD_OK),
    }
    print("\n⑤ 判定ロジック")
    for name, ok in checks.items():
        print(f"  [{'OK  ' if ok else 'NG  '}] {name}")
    print(f"\n  {judge(checks)}")
    print("\n  この依頼に「効果は +X でした」と答えてはいけない。書くべきはこう:")
    print("    ・素朴な差 "
          f"{naive.estimate:+.3f} は交絡を含んでおり、因果効果ではない")
    print("    ・交絡因子が記録されていれば "
          f"{ipw.estimate:+.3f}（95%CI [{ipw.ci[0]:+.3f}, {ipw.ci[1]:+.3f}]）まで戻せるが、")
    print("      これは『記録されていない交絡は無い』という**検証できない仮定**の上の数字")
    print("\n  答えるために追加で要るデータ:")
    for item in ("処置（施策の適用）が乱数で割り付けられた期間、あるいは A/B の記録",
                 "処置の直前に測った利用意欲・過去購買・流入経路などの共変量",
                 "処置のタイミングがずれた自然実験（DID が使えるパネル）",
                 "処置の割付だけに効いて結果には効かない変数（操作変数）",
                 "処置確率が閾値で不連続に変わる運用ルール（RDD が使える）"):
        print(f"    - {item}")

    draw([
        ("素朴な群間差", naive.estimate, naive.ci[0], naive.ci[1]),
        ("共変量調整（z 観測）", float(ols.b[1]), float(lo_ols), float(hi_ols)),
        ("IPW（z 観測）", ipw.estimate, ipw.ci[0], ipw.ci[1]),
    ])


if __name__ == "__main__":
    main()
