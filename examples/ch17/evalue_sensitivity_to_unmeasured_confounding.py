"""未観測の交絡は証明できない。ならば「どれだけ強ければ結論が消えるか」を数字にする。

観察研究の結論には必ず「未観測の交絡があるかもしれない」という留保がつく。無いことは
証明できないので、留保は永遠に消えない。E値はここで発想を裏返す——
**この関連を丸ごと説明し尽くすには、未観測の交絡はどれだけ強くなければならないか。**

必要な強さが「喫煙と肺がん」級なら結論は頑健だし、「たいていの生活習慣」級なら
危うい。計算は電卓でできる。$\\mathrm{RR}$ が観測された相対リスクなら、
E値は $\\mathrm{RR} + \\sqrt{\\mathrm{RR}(\\mathrm{RR}-1)}$ でしかない。

    uv run python examples/ch17/evalue_sensitivity_to_unmeasured_confounding.py
"""

import numpy as np

from toukei_tashikame import causal, plots

RR_OBS, RR_LO, RR_HI = 2.40, 1.62, 3.55   # 観測されたリスク比と 95%CI


def e_value_by_hand(rr: float) -> float:
    """E値。バイアス因子 B = ab/(a+b-1) が RR に等しくなる a=b を解いた答え。"""
    rr = 1.0 / rr if rr < 1 else rr
    return rr + np.sqrt(rr * (rr - 1.0))


def bias_factor(rr_tu, rr_uy):
    """未観測交絡 U がもたらしうる最大の見かけ上の比（VanderWeele-Ding）。"""
    return (rr_tu * rr_uy) / (rr_tu + rr_uy - 1.0)


def needed_partner(rr_target: float, rr_tu: float) -> float:
    """片方の強さを固定したとき、結論を消すのに要るもう片方の強さ。"""
    # ab/(a+b-1) = rr_target を b について解く。分母が 0 以下なら解なし。
    denom = rr_tu - rr_target
    return float("inf") if denom <= 0 else rr_target * (rr_tu - 1.0) / denom


def draw(grid, ev_point, ev_ci) -> None:
    fig, ax = plots.figure(w=1.4)
    pal = plots.PALETTE
    a, b = np.meshgrid(grid, grid)
    rr_true = RR_OBS / bias_factor(a, b)

    cs = ax.contourf(a, b, rr_true, levels=[0.0, 1.0, 1.2, 1.5, 2.0, 2.4, 10.0],
                     colors=["#f4d9c4", "#e9e9e4", "#dcdcd6", "#cccdc6", "#bcbdb6", "#adaea7"])
    ax.clabel(ax.contour(a, b, rr_true, levels=[1.2, 1.5, 2.0], colors=pal["ink2"],
                         linewidths=0.6), fontsize=5.6, fmt="%.1f")
    ax.contour(a, b, rr_true, levels=[1.0], colors=[pal["truth"]], linewidths=1.3)
    ax.contour(a, b, RR_LO / bias_factor(a, b), levels=[1.0], colors=[pal["reject"]],
               linewidths=1.1, linestyles="--")
    ax.plot(grid, grid, color=pal["ink2"], lw=0.6, ls=":")
    ax.scatter([ev_point], [ev_point], s=22, color=pal["truth"], zorder=6, lw=0)
    ax.annotate(f"E値 {ev_point:.2f}\n（点推定が消える）", xy=(ev_point, ev_point),
                fontsize=6.0, color=pal["truth"], ha="left", va="bottom",
                xytext=(4, 3), textcoords="offset points")
    ax.scatter([ev_ci], [ev_ci], s=22, color=pal["reject"], zorder=6, lw=0)
    ax.annotate(f"E値 {ev_ci:.2f}\n（CI下限が消える）", xy=(ev_ci, ev_ci), fontsize=6.0,
                color=pal["reject"], ha="right", va="top",
                xytext=(-3, -3), textcoords="offset points")
    fig.colorbar(cs, ax=ax).set_label("交絡を除いた後に残る真の RR", fontsize=6.2)
    ax.set_xlabel("処置 → 未観測交絡 U の強さ（RR）")
    ax.set_ylabel("U → 結果 の強さ（RR）")
    fig.tight_layout()
    plots.save(fig, "fig-17-11-evalue-contour.png")


def main() -> None:
    plots.setup()
    ev_point = e_value_by_hand(RR_OBS)
    ev_ci = e_value_by_hand(RR_LO)

    print(f"--- 観測された関連: RR = {RR_OBS:.2f}"
          f"（95%CI [{RR_LO:.2f}, {RR_HI:.2f}]）---\n")
    print(f"  点推定の E値      {ev_point:.2f}   = {RR_OBS:.2f} + √({RR_OBS:.2f}×"
          f"{RR_OBS - 1:.2f})")
    print(f"  CI 下限の E値     {ev_ci:.2f}   = {RR_LO:.2f} + √({RR_LO:.2f}×"
          f"{RR_LO - 1:.2f})")
    print(f"  照合: causal.e_value = {causal.e_value(RR_OBS):.4f} / "
          f"{causal.e_value(RR_OBS, RR_LO):.4f}")

    print(f"\n  読み方: 未観測の交絡 U が、処置とも結果とも RR {ev_point:.2f} 以上で"
          "結びついていれば、")
    print(f"  観測された {RR_OBS:.2f} 倍は交絡だけで説明がつく。逆に、そこまで強い U が"
          "考えにくいなら、")
    print(f"  関連の一部は残る。{ev_ci:.2f} 以上なら、CI 下限すら 1 をまたぐ"
          "——「有意」も消える。")

    print("\n--- 片方の強さを決めたとき、もう片方はどれだけ要るか（点推定を消すには）---\n")
    print(f"{'処置 → U の RR':>16}{'必要な U → 結果の RR':>22}")
    for rr_tu in sorted([1.5, 2.0, 2.4, 3.0, ev_point, 5.0, 10.0]):
        need = needed_partner(RR_OBS, rr_tu)
        text = "どれだけ強くても不可能" if np.isinf(need) else f"{need:.2f}"
        mark = "  ← E値（両方が同じ強さの点）" if abs(rr_tu - ev_point) < 1e-9 else ""
        print(f"{rr_tu:>16.2f}{text:>22}{mark}")
    print(f"\n  片方が {RR_OBS:.2f} 以下だと、相方をいくら強くしても説明し切れない。")
    print("  バイアス因子 ab/(a+b−1) は、片方を固定すると頭打ちになるからである。")
    print(f"  E値 {ev_point:.2f} は「両方が同じ強さ」という最も楽な場合の値で、"
          "必要な強さの下限にあたる。")

    print("\n--- 交絡の強さの目安（既知の関連との比較）---\n")
    for label, rr in (("既知の共変量（年齢層）", 1.4), ("よくある生活習慣の指標", 2.0),
                      ("喫煙と心疾患", 2.5), ("喫煙と肺がん", 10.0)):
        verdict = "消せる" if bias_factor(rr, rr) >= RR_OBS else "消せない"
        print(f"{label:<26}RR {rr:>5.1f}   バイアス因子 "
              f"{bias_factor(rr, rr):>5.2f}   結論を{verdict}")
    print(f"\n  この {RR_OBS:.2f} 倍を消すには「喫煙と肺がん」に迫る強さの未観測交絡が要る、"
          "とまでは言えない。")
    print(f"  必要なのは RR {ev_point:.2f} 前後——つまり、"
          "測り忘れた強い変数が1つあれば十分ありうる水準である。")
    print("  E値は結論を守る道具でも壊す道具でもない。"
          "「どれくらいの未観測交絡なら耐えられるか」を")
    print("  読者と共有するための共通の物差しで、"
          "小さければ小さいなりに、正直に書くために使う。")

    print("\n  注意: E値が大きいことは「交絡が無い」ことを意味しない。"
          "測っていないものについて、")
    print("  データが語れることは何も無い。E値が答えているのは"
          "「もし在るとしたら、どれだけ強い必要があるか」だけである。")

    draw(np.linspace(1.0, 8.0, 240), ev_point, ev_ci)


if __name__ == "__main__":
    main()
