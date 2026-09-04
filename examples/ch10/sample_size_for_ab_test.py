"""A/Bテストのサンプルサイズ設計 — 「CVR 3.0% を 3.3% に上げたい」を n に翻訳する。

要件は4つの数字で決まる。ベースラインの CVR、拾いたい改善幅、α、目標検出力。この4つを
入れると n が1つ出てくる。逆に言えば、n を決めずに始めた A/B テストは「何を拾えるか」を
決めずに始めたということで、有意にならなかったときに何も言えない。

出てくる n が現実的でないことのほうが多い。相対10%の改善（3.0%→3.3%）を拾うには
5万件/群が要る。これは設計の失敗ではなく、**小さい改善は小さい標本では見えない**という
事実そのもので、要件のほうを見直す材料になる。

設計した n が本当に検出力 0.8 を出すかは、回して数えれば確かめられる。

    uv run python examples/ch10/sample_size_for_ab_test.py
"""

import numpy as np
from scipy import stats

from toukei_tashikame import datasets, plots, power, sim, testing

P_A, ALPHA, TARGET_POWER = 0.030, 0.05, 0.80
PV_PER_DAY = 10_000            # 1日あたりの総PV。2群に半分ずつ振る
TRIALS_LOG, TRIALS_FAST, SEED = 2_000, 10_000, 104


def z_test_power(n: int, p_a: float, p_b: float, trials: int, rng) -> float:
    """成功数を二項分布から直接引いて、プールした比率の z 検定を一気にかける。

    1件ずつのログを作っても成功数の分布は同じなので、10,000回まわすときはこちらを使う。
    """
    ka = rng.binomial(n, p_a, size=trials)
    kb = rng.binomial(n, p_b, size=trials)
    pooled = (ka + kb) / (2 * n)
    se = np.sqrt(pooled * (1 - pooled) * 2 / n)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (kb / n - ka / n) / se
    p = 2 * stats.norm.sf(np.abs(np.nan_to_num(z)))
    return float((p < ALPHA).mean())


def main() -> None:
    plots.setup()
    rng = np.random.default_rng(SEED)

    for lift in (0.10, 0.20):
        p_b = P_A * (1 + lift)
        n = power.n_for_proportions(P_A, p_b, power=TARGET_POWER, alpha=ALPHA)
        days = 2 * n / PV_PER_DAY
        print(f"--- CVR {P_A:.1%} → {p_b:.1%}（相対 {lift:.0%} 改善）を"
              f"検出力 {TARGET_POWER} で拾う ---")
        print(f"  必要な n = {n:,} / 群（総数 {2 * n:,}）")
        print(f"  1日 {PV_PER_DAY:,} PV を2群に振るなら {days:.1f} 日")
        print(f"  実測検出力（二項で {TRIALS_FAST:,} 回）"
              f"  {z_test_power(n, P_A, p_b, TRIALS_FAST, rng):.4f}")
        print()

    # --- 設計した n を、1件ずつのログを作る datasets.ab_test で確かめる ---
    n = power.n_for_proportions(P_A, P_A * 1.10, power=TARGET_POWER, alpha=ALPHA)

    def one_trial(trial_rng):
        d = datasets.ab_test(n_a=n, n_b=n, p_a=P_A, lift=0.10,
                             seed=int(trial_rng.integers(2**32)))
        return testing.prop_2samp(int(d.a.sum()), n, int(d.b.sum()), n).pvalue

    with sim.Timer("datasets.ab_test で 2,000 回"):
        res = sim.rejection_rate(one_trial, alpha=ALPHA, trials=TRIALS_LOG,
                                 seed=SEED, progress=False)
    print(f"--- 1件ずつのログ（datasets.ab_test, n={n:,}/群）を {TRIALS_LOG:,} 回 ---")
    print(f"  実測検出力 {res.rate:.4f} ± {1.96 * res.se:.4f}"
          f"  ← 設計値 {TARGET_POWER} を包んでいる")
    sample = datasets.ab_test(n_a=n, n_b=n, p_a=P_A, lift=0.10, seed=SEED)
    print(f"  1回ぶんの中身: A の CVR {sample.a.mean():.4%} / B の CVR {sample.b.mean():.4%}"
          f"（真値 {sample.p_a:.1%} と {sample.p_b:.1%}）")

    # --- 図。拾いたい改善幅を横軸に、必要な n を縦軸に ---
    lifts = np.linspace(0.02, 0.50, 60)
    ns = [power.n_for_proportions(P_A, P_A * (1 + lf), power=TARGET_POWER, alpha=ALPHA)
          for lf in lifts]
    fig, ax = plots.figure(w=1.15, h=1.05)
    ax.plot(100 * lifts, ns, color=plots.PALETTE["estimate"], lw=1.3, zorder=3)
    for lf, note in ((0.10, "相対10%"), (0.20, "相対20%")):
        nn = power.n_for_proportions(P_A, P_A * (1 + lf), power=TARGET_POWER, alpha=ALPHA)
        ax.scatter([100 * lf], [nn], s=14, color=plots.PALETTE["truth"], zorder=5)
        ax.annotate(f"{note}\n{nn:,}/群\n{2 * nn / PV_PER_DAY:.1f}日",
                    xy=(100 * lf, nn), xytext=(6, 2), textcoords="offset points",
                    fontsize=6.0, color=plots.PALETTE["truth"])
    ax.set_yscale("log")
    ax.set_xlabel(f"拾いたい相対改善幅（%、ベースライン CVR {P_A:.1%}）")
    ax.set_ylabel("必要な n / 群（対数目盛）")
    ax.set_title(f"α={ALPHA} 両側、検出力 {TARGET_POWER}")
    fig.tight_layout()
    plots.save(fig, "fig-10-4-sample-size-ab.png")


if __name__ == "__main__":
    main()
