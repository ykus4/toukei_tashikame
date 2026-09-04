"""崩し — 「n はいくつか」は行数ではなくクラスタ数で決まる。

1人のユーザーから10回の計測を取れば行数は10倍になるが、独立な情報が10倍になるわけでは
ない。同じ人の10回は互いに似ているからで、その似ぐあいを表すのが級内相関 ICC である。
それでも行を独立と思って t 検定にかけると、標準誤差が実際より小さく出て、第一種の誤りが
名目の α を大きく超える。

ICC = 0.0 / 0.1 / 0.3 / 0.5 の4水準で、50ユーザー × 10計測 × 2群のデータを 10,000 組
作り、(a) 行をそのまま独立扱いした t 検定と、(b) ユーザー平均に潰してからの t 検定を
比べる。潰すと行数は 1000 から 100 に減るが、正しいのは減ったほうである。

    uv run python examples/ch08/clustered_data_inflates_type1.py
"""

import numpy as np
from scipy import stats

from toukei_tashikame import plots

N_USER, PER_USER = 50, 10      # 1群あたり50ユーザー、1ユーザー10計測
TRIALS, SEED, ALPHA = 10_000, 811, 0.05
ICCS = (0.0, 0.1, 0.3, 0.5)


def draw(rng, icc: float):
    """帰無（2群に差なし）のクラスタデータを ``TRIALS`` 組作る。

    形は ``(TRIALS, N_USER, PER_USER)``。ユーザー効果は10計測で共有され、その分散が
    全体の ICC の割合を占める。``datasets.clustered`` と同じ作り方である。
    """
    shape = (TRIALS, N_USER, 1)
    user = rng.normal(0.0, np.sqrt(icc), size=shape)
    within = rng.normal(0.0, np.sqrt(1.0 - icc), size=(TRIALS, N_USER, PER_USER))
    return user + within


def se(rate: float) -> float:
    return float(np.sqrt(rate * (1 - rate) / TRIALS))


def main() -> None:
    plots.setup()
    print(f"帰無が真（2群に差なし）。1群 {N_USER}ユーザー × {PER_USER}計測 = "
          f"{N_USER * PER_USER}行、{TRIALS:,}組、α={ALPHA}")
    print(f"\n{'ICC':>6}{'行を独立扱い':>16}{'ユーザー平均に潰す':>20}{'倍率':>8}")

    naive, collapsed = [], []
    for i, icc in enumerate(ICCS):
        rng = np.random.default_rng(SEED + 100 * i)
        a, b = draw(rng, icc), draw(rng, icc)

        # (a) 1000行 vs 1000行。同じユーザーの10行を別々の観測だと思っている
        p_naive = stats.ttest_ind(a.reshape(TRIALS, -1), b.reshape(TRIALS, -1),
                                  axis=1, equal_var=False).pvalue
        # (b) ユーザー平均に潰す。50 vs 50 になるが、これが独立な単位の数
        p_mean = stats.ttest_ind(a.mean(axis=2), b.mean(axis=2),
                                 axis=1, equal_var=False).pvalue

        rn = float((p_naive < ALPHA).mean())
        rc = float((p_mean < ALPHA).mean())
        naive.append(rn)
        collapsed.append(rc)
        print(f"{icc:>6.1f}{rn:>16.4f}{rc:>20.4f}{rn / ALPHA:>7.1f}倍")

    print(f"\n  ICC=0.0（本当に独立）なら独立扱いでも {naive[0]:.4f} ± "
          f"{1.96 * se(naive[0]):.4f} で問題ない")
    print(f"  ICC=0.3 では {naive[2]:.4f} ± {1.96 * se(naive[2]):.4f}"
          f"（名目の {naive[2] / ALPHA:.1f}倍）。ユーザー平均に潰せば {collapsed[2]:.4f} に戻る")
    print(f"  ICC=0.5 では {naive[3]:.4f}。差がまったく無いのに、"
          "10回に4回は「有意差あり」と言ってしまう")
    print("\n  行数を増やしても独立な情報は増えない。増えるのは「増えたつもり」だけである。"
          "設計時に数えるべきは行数ではなくユーザー数（クラスタ数）")

    # 有効標本サイズ（デザイン効果）。理論値と照らす
    print(f"\n--- デザイン効果 1 + (m-1)·ICC （m={PER_USER}）---")
    for icc, rn in zip(ICCS, naive, strict=True):
        deff = 1 + (PER_USER - 1) * icc
        print(f"  ICC={icc:.1f}  deff={deff:.2f}  有効な n = "
              f"{N_USER * PER_USER / deff:.0f}行相当（実際の行数 {N_USER * PER_USER}）"
              f"   実測の第一種の誤り {rn:.4f}")

    # --- 図 ---
    fig, ax = plots.figure(w=1.25)
    ax.plot(ICCS, naive, marker="o", ms=3.5, lw=1.3, color=plots.PALETTE["reject"], zorder=4)
    ax.plot(ICCS, collapsed, marker="o", ms=3.5, lw=1.3, color=plots.PALETTE["estimate"],
            zorder=4)
    ax.annotate("行をそのまま独立扱い", xy=(ICCS[-1], naive[-1]), xytext=(-4, -8),
                textcoords="offset points", ha="right", va="top", fontsize=6.2,
                color=plots.PALETTE["reject"])
    ax.annotate("ユーザー平均に潰す", xy=(ICCS[1], collapsed[1]), xytext=(0, -8),
                textcoords="offset points", ha="center", va="top", fontsize=6.2,
                color=plots.PALETTE["estimate"])
    for x, y in zip(ICCS, naive, strict=True):
        ax.annotate(f"{y:.4f}", xy=(x, y), xytext=(0, 5), textcoords="offset points",
                    ha="center", fontsize=5.8, color=plots.PALETTE["reject"])
    plots.mark_truth(ax, ALPHA, "名目 α = 0.05", axis="y")
    ax.set_xlabel("級内相関 ICC")
    ax.set_ylabel("第一種の誤り")
    ax.set_ylim(0, max(naive) * 1.18)
    ax.set_title(f"{N_USER}ユーザー×{PER_USER}計測×2群、{TRIALS:,}回")
    fig.tight_layout()
    plots.save(fig, "fig-8-11-clustered-type1.png")


if __name__ == "__main__":
    main()
