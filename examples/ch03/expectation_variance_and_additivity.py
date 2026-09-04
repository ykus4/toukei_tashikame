"""期待値と分散を数え上げ、分散の加法性が独立でないと崩れることを測る。

期待値 $\\mathbb{E}[X]$ は「たくさん引いたときの平均が寄っていく先」、分散はその
散らばり。まずサイコロ1個を100万回振って、紙の上で解いた 3.5 と 35/12 に本当に
寄ることを見る。

そのうえで、よく使われるのに条件を忘れられがちな式を1つ壊す。
$\\operatorname{Var}(X+Y) = \\operatorname{Var}(X) + \\operatorname{Var}(Y)$ は
**独立（無相関）のときにだけ**成り立つ。正しくは $+\\,2\\operatorname{Cov}(X,Y)$ が
付く。相関 0.5 や 0.8 の変数を足して分散を数えると、独立を仮定した見積もりが
どれだけ小さすぎるかが人数ならぬ数値で出る。

    uv run python examples/ch03/expectation_variance_and_additivity.py
"""

import numpy as np

from toukei_tashikame import datasets, describe, plots

N_DICE = 1_000_000
N_PAIR = 100_000
SEED_DICE = 9
SEED_PAIR = 10
RHOS = (0.0, 0.5, 0.8)


def main() -> None:
    plots.setup()

    print(f"--- 1. サイコロ1個を {N_DICE:,} 回（seed={SEED_DICE}）---")
    rng = np.random.default_rng(SEED_DICE)
    x = rng.integers(1, 7, size=N_DICE).astype(float)

    faces = np.arange(1.0, 7.0)
    e_theory = faces.mean()                       # Σ x·(1/6) = 3.5
    v_theory = ((faces - e_theory) ** 2).mean()   # 35/12
    print(f"  {'':<14}{'実測':>10}{'理論':>10}{'差':>10}")
    print(f"  {'E[X]':<14}{x.mean():>10.4f}{e_theory:>10.4f}{x.mean() - e_theory:>10.4f}")
    print(f"  {'Var(X)':<14}{describe.var(x):>10.4f}{v_theory:>10.4f}"
          f"{describe.var(x) - v_theory:>10.4f}")
    print(f"  {'SD(X)':<14}{describe.sd(x):>10.4f}{np.sqrt(v_theory):>10.4f}"
          f"{describe.sd(x) - np.sqrt(v_theory):>10.4f}")
    print(f"  理論の分散は 35/12 = {35 / 12:.4f}。100万回でも小数第3位までしか合わない")

    print("\n  平均の寄り方（先頭 n 個までの平均）")
    for n in [10, 10**2, 10**3, 10**4, 10**5, 10**6]:
        print(f"    n={n:>9,}  {x[:n].mean():.4f}")

    print("\n--- 2. Var(X+Y) = Var(X) + Var(Y) は独立のときだけ ---")
    print(f"  分散1の X, Y を相関を変えて {N_PAIR:,} 組ずつ（seed={SEED_PAIR}）")
    print(f"  {'相関ρ':>6}{'Var(X)':>9}{'Var(Y)':>9}{'Var(X+Y)':>11}"
          f"{'独立仮定':>10}{'理論2+2ρ':>10}{'倍率':>8}")
    measured = []
    for rho in RHOS:
        xx, yy = datasets.bivariate_normal(N_PAIR, rho=rho, seed=SEED_PAIR)
        vx, vy = describe.var(xx), describe.var(yy)
        vsum = describe.var(xx + yy)
        measured.append(vsum)
        print(f"  {rho:>6.1f}{vx:>9.4f}{vy:>9.4f}{vsum:>11.4f}"
              f"{vx + vy:>10.4f}{2 + 2 * rho:>10.4f}{vsum / (vx + vy):>8.2f}倍")

    print("\n  足りない分は 2·Cov(X, Y)。共分散を測って足し直すと合う:")
    for rho in RHOS:
        xx, yy = datasets.bivariate_normal(N_PAIR, rho=rho, seed=SEED_PAIR)
        cov = float(np.cov(xx, yy, ddof=1)[0, 1])
        rebuilt = describe.var(xx) + describe.var(yy) + 2 * cov
        print(f"    ρ={rho:.1f}  Var(X)+Var(Y)+2Cov = {rebuilt:.4f}  "
              f"（実測 {describe.var(xx + yy):.4f}、差 {abs(rebuilt - describe.var(xx + yy)):.2e}）")

    print("\n  ρ=0.8 では独立を仮定した見積もりが実際の "
          f"{(2 + 2 * 0.8) / 2.0:.2f} 分の1。標準誤差なら "
          f"{np.sqrt(2.0 / (2 + 2 * 0.8)):.2f} 倍に見積もる。"
          "相関したデータで信頼区間が狭すぎるのはこれが理由")

    fig, (ax1, ax2) = plots.figure(1, 2, w=2.0, h=0.95)

    # 左: サイコロの目の相対頻度と、期待値の位置。
    counts = np.bincount(x.astype(int), minlength=7)[1:] / N_DICE
    ax1.bar(faces, counts, width=0.7, color=plots.PALETTE["data"], alpha=0.65, lw=0)
    ax1.axhline(1 / 6, color=plots.PALETTE["estimate"], lw=0.9, ls="--", dashes=(4, 2.0))
    plots.mark_truth(ax1, e_theory, f"E[X] = {e_theory}")
    ax1.annotate(f"実測平均 {x.mean():.4f}", xy=(0.03, 0.06), xycoords="axes fraction",
                 fontsize=6.2, color=plots.PALETTE["estimate"])
    ax1.set_ylim(0, 0.22)
    ax1.set_xlabel("出た目")
    ax1.set_ylabel("相対頻度")
    ax1.set_title(f"サイコロ {N_DICE:,} 回")

    # 右: Var(X+Y) の実測と、独立を仮定した見積もり。
    pos = np.arange(len(RHOS))
    ax2.bar(pos, measured, width=0.5, color=plots.PALETTE["estimate"], alpha=0.85, lw=0)
    plots.mark_truth(ax2, 2.0, "独立を仮定すると 2.0", axis="y")
    for p, v in zip(pos, measured, strict=True):
        ax2.annotate(f"{v:.3f}", xy=(p, v), xytext=(0, 2), textcoords="offset points",
                     ha="center", fontsize=6.2, color=plots.PALETTE["estimate"])
    ax2.set_xticks(pos)
    ax2.set_xticklabels([f"ρ={r}" for r in RHOS])
    ax2.set_ylabel("Var(X + Y)")
    ax2.set_ylim(0, max(measured) * 1.25)
    ax2.set_title("加法性は独立の上にしか立たない")

    fig.tight_layout()
    plots.save(fig, "fig-3-7-variance-additivity.png")


if __name__ == "__main__":
    main()
