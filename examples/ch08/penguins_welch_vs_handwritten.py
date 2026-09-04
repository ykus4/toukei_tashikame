"""Welch の t 検定を手で組み立て、scipy と一致することを実データで確かめる。

検定は「仮定 → 統計量 → 分布 → 尾側確率」の4段でしかない。その4段を Palmer Penguins の
Adelie（n=151）と Gentoo（n=123）のくちばし長で自分で書き、``ttest_ind(equal_var=False)``
と突き合わせる。一致すれば、以降 scipy を呼ぶときに「中で何が起きているか」を知った上で
呼べる。Welch の自由度が整数にならないところがこの近似の顔である。

    uv run python examples/ch08/penguins_welch_vs_handwritten.py
"""

import numpy as np
from scipy import stats

from toukei_tashikame import datasets, plots, testing


def welch_by_hand(x, y):
    """Welch の t 統計量・自由度・両側 p 値を、式のとおりに組み立てる。"""
    n1, n2 = x.size, y.size
    v1, v2 = x.var(ddof=1), y.var(ddof=1)

    # 1. 統計量 — 平均差を、各群の分散から作った標準誤差で割る（プールしない）
    se = np.sqrt(v1 / n1 + v2 / n2)
    t = (x.mean() - y.mean()) / se

    # 2. 分布 — Welch–Satterthwaite の近似自由度。整数にならない
    df = (v1 / n1 + v2 / n2) ** 2 / ((v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1))

    # 3. 尾側確率 — t 分布の両裾
    p = 2 * stats.t.sf(abs(t), df)
    return float(t), float(df), float(p), float(se)


def main() -> None:
    plots.setup()

    df = datasets.penguins()
    adelie = df.loc[df["species"] == "Adelie", "bill_length_mm"].to_numpy()
    gentoo = df.loc[df["species"] == "Gentoo", "bill_length_mm"].to_numpy()

    print("--- データ（くちばし長 mm）---")
    for name, g in (("Adelie", adelie), ("Gentoo", gentoo)):
        print(f"  {name:<7} n={g.size:3d}  平均={g.mean():.4f}  標準偏差={g.std(ddof=1):.4f}")

    t_hand, df_hand, p_hand, se = welch_by_hand(adelie, gentoo)
    print("\n--- 手で書いた Welch ---")
    print(f"  標準誤差 se = {se:.6f}")
    print(f"  t  = {t_hand:.4f}")
    print(f"  df = {df_hand:.2f}   ← 整数でない。これが Welch–Satterthwaite 近似の顔")
    print(f"  p  = {p_hand:.4g}")

    sp = stats.ttest_ind(adelie, gentoo, equal_var=False)
    print("\n--- scipy の ttest_ind(equal_var=False) ---")
    print(f"  t  = {sp.statistic:.4f}")
    print(f"  df = {sp.df:.2f}")
    print(f"  p  = {sp.pvalue:.4g}")
    gaps = [abs(t_hand - sp.statistic), abs(df_hand - sp.df), abs(p_hand - sp.pvalue)]
    print(f"\n  手書きと scipy の最大差 = {max(gaps):.3e}"
          "   ← 同じ式を同じ順序で計算しているので、丸め誤差すら出ない")

    print("\n--- 参考: 等分散を仮定した Student 版（本文 8-3 の比較）---")
    st = testing.t_ind(adelie, gentoo, equal_var=True)
    print(f"  {st}")
    print(f"\n  分散比 = {adelie.var(ddof=1) / gentoo.var(ddof=1):.4f}"
          "  ← ここが1から離れるほど Student と Welch は食い違う")

    # --- 図 ---
    fig, (ax1, ax2) = plots.figure(1, 2, w=1.5)

    bins = np.linspace(30, 60, 40)
    ax1.hist(adelie, bins=bins, color=plots.PALETTE["data"], alpha=0.55, lw=0)
    ax1.hist(gentoo, bins=bins, color=plots.PALETTE["alt"], alpha=0.55, lw=0)
    ax1.axvline(adelie.mean(), color=plots.PALETTE["data"], lw=1.2)
    ax1.axvline(gentoo.mean(), color=plots.PALETTE["alt"], lw=1.2)
    ax1.annotate(f"Adelie\n平均 {adelie.mean():.1f}", xy=(adelie.mean(), 0.97),
                 xycoords=("data", "axes fraction"), ha="right", va="top",
                 fontsize=6.0, color=plots.PALETTE["data"])
    ax1.annotate(f"Gentoo\n平均 {gentoo.mean():.1f}", xy=(gentoo.mean(), 0.97),
                 xycoords=("data", "axes fraction"), ha="left", va="top",
                 fontsize=6.0, color=plots.PALETTE["alt"])
    ax1.set_xlabel("くちばし長 (mm)")
    ax1.set_ylabel("度数")
    ax1.set_title("2群の分布")

    diff = adelie.mean() - gentoo.mean()
    half = stats.t.ppf(0.975, df_hand) * se
    ax2.set_xlim(diff - 6 * half, 2 * half)
    ax2.set_ylim(0, 1)
    plots.mark_interval(ax2, diff - half, diff + half, y=0.5)
    ax2.plot([diff], [0.5], "o", ms=3.5, color=plots.PALETTE["estimate"], zorder=5)
    ax2.annotate(f"平均差 {diff:.2f}\n95%CI [{diff - half:.2f}, {diff + half:.2f}]",
                 xy=(diff, 0.5), xytext=(0, 8), textcoords="offset points",
                 ha="center", va="bottom", fontsize=6.0, color=plots.PALETTE["estimate"])
    plots.mark_truth(ax2, 0.0, "帰無仮説 差=0")
    ax2.set_yticks([])
    ax2.set_xlabel("Adelie − Gentoo のくちばし長差 (mm)")
    ax2.set_title(f"Welch の95%信頼区間（df={df_hand:.1f}）")

    fig.tight_layout()
    plots.save(fig, "fig-8-3-penguins-welch.png")


if __name__ == "__main__":
    main()
