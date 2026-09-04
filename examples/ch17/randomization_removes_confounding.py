"""ランダム化は何を消しているのか — コイン投げの前後で 10,000 回ずつ数える。

同じ生成規則の世界を2つ用意する。片方は交絡変数 $Z$ が処置の割付に効き、もう片方は
コイン投げで割り付ける。$Z$ が結果 $Y$ に効くことは両方で同じで、違うのは
「$Z$ が $T$ に矢印を持つかどうか」だけである。

その1本の矢印を抜くと、素朴な群間差がそのまま ATE の不偏推定になり、95%区間の被覆が
0.95 に戻る。ランダム化が消しているのは交絡変数そのものではなく、
**交絡変数から処置への矢印**である。$Z$ は最後まで $Y$ に効き続けている。

    uv run python examples/ch17/randomization_removes_confounding.py
"""

import numpy as np
from scipy import stats

from toukei_tashikame import causal, plots, sim

ATE = 2.5          # 真の平均処置効果
N = 200            # 1回の実験の被験者数
GAMMA = 1.2        # Z → T の強さ（ランダム化した世界では 0 になる）
BETA_Z = 1.5       # Z → Y の強さ（両方の世界で同じ）
TRIALS, SEED = 10_000, 174


def one_experiment(rng, randomized: bool):
    """1回ぶんの実験。素朴な群間差と、その 95% 区間（Welch）を返す。"""
    z = rng.normal(0.0, 1.0, size=N)            # 交絡変数（例: もともとの熱心さ）
    if randomized:
        t = (rng.random(N) < 0.5).astype(int)   # コイン投げ。Z を一切見ない
    else:
        t = (rng.random(N) < 1 / (1 + np.exp(-GAMMA * z))).astype(int)
    y = 5.0 + ATE * t + BETA_Z * z + rng.normal(0.0, 1.0, size=N)

    y1, y0 = y[t == 1], y[t == 0]
    if y1.size < 2 or y0.size < 2:
        return np.nan, np.nan
    est = y1.mean() - y0.mean()
    se = np.sqrt(y1.var(ddof=1) / y1.size + y0.var(ddof=1) / y0.size)
    df = (se**2) ** 2 / ((y1.var(ddof=1) / y1.size) ** 2 / (y1.size - 1)
                         + (y0.var(ddof=1) / y0.size) ** 2 / (y0.size - 1))
    h = stats.t.ppf(0.975, df) * se
    return est - h, est + h


def balance(rng, randomized: bool):
    """1回の大きめの実験で、群間の共変量バランス（標準化平均差）を見る。"""
    n = 2000
    z = rng.normal(0.0, 1.0, size=n)
    w = rng.normal(0.0, 1.0, size=n)            # 結果にも処置にも効かないダミー共変量
    if randomized:
        t = (rng.random(n) < 0.5).astype(int)
    else:
        t = (rng.random(n) < 1 / (1 + np.exp(-GAMMA * z))).astype(int)
    return causal.balance_table(np.column_stack([z, w]), t, names=["Z（交絡変数）", "W（無関係）"])


def draw(res_rand, res_conf) -> None:
    fig, ax = plots.figure(w=1.4)
    pal = plots.PALETTE
    est_rand = res_rand.intervals.mean(axis=1)
    est_conf = res_conf.intervals.mean(axis=1)
    bins = np.linspace(min(est_rand.min(), est_conf.min()),
                       max(est_rand.max(), est_conf.max()), 70)
    ax.hist(est_rand, bins=bins, color=pal["estimate"], alpha=0.55, lw=0)
    ax.hist(est_conf, bins=bins, color=pal["reject"], alpha=0.55, lw=0)
    plots.mark_truth(ax, ATE, f"真の ATE = {ATE}")
    ax.annotate(f"ランダム化あり\n平均 {est_rand.mean():.3f}", xy=(est_rand.mean(), 0.62),
                xycoords=("data", "axes fraction"), ha="center", fontsize=6.2,
                color=pal["estimate"])
    ax.annotate(f"ランダム化なし\n平均 {est_conf.mean():.3f}", xy=(est_conf.mean(), 0.62),
                xycoords=("data", "axes fraction"), ha="center", fontsize=6.2,
                color=pal["reject"])
    ax.set_xlabel(f"素朴な群間差の推定値（n={N} の実験を {TRIALS:,} 回）")
    ax.set_ylabel("回数")
    fig.tight_layout()
    plots.save(fig, "fig-17-4-randomization-balance.png")


def main() -> None:
    plots.setup()

    res_rand = sim.coverage(lambda r: one_experiment(r, randomized=True), truth=ATE,
                            trials=TRIALS, seed=SEED, progress=False)
    res_conf = sim.coverage(lambda r: one_experiment(r, randomized=False), truth=ATE,
                            trials=TRIALS, seed=SEED, progress=False)

    print(f"--- 同じ世界を2通りに割り付けて {TRIALS:,} 回ずつ（n={N}, 真の ATE = {ATE}）---\n")
    print(f"{'割付':<20}{'平均推定':>10}{'バイアス':>10}{'推定のSD':>10}{'95%区間の被覆':>14}")
    for label, res in (("ランダム化あり（コイン）", res_rand), ("ランダム化なし（Z依存）", res_conf)):
        est = res.intervals.mean(axis=1)
        print(f"{label:<20}{est.mean():>10.3f}{est.mean() - ATE:>+10.3f}"
              f"{est.std(ddof=1):>10.3f}{res.rate:>12.4f} ± {1.96 * res.se:.4f}")
    print(f"\n  ランダム化なしのバイアス {res_conf.intervals.mean() - ATE:+.3f} は"
          f"「たまたま」ではない。{TRIALS:,} 回の平均でも消えない。")
    print("  区間を狭めても被覆は戻らない。n を増やせば区間は縮み、真値を外す確率はむしろ上がる。")
    print("  バイアスは分散と違って、標本を増やしても消えない種類の誤りである。")

    rng = np.random.default_rng(SEED + 1)
    print("\n--- 割付直後の共変量バランス（n=2,000 の1回ぶん、|SMD|<0.1 が目安）---")
    for label, randomized in (("ランダム化あり", True), ("ランダム化なし", False)):
        tab = balance(np.random.default_rng(rng.integers(1 << 31)), randomized)
        print(f"\n  [{label}]")
        print(tab.round(4).to_string())
    print("\n  ランダム化した世界では、Z も W も——そして測っていない何もかもも——揃う。")
    print("  揃うことを保証しているのは割付の手続きであって、事後の調整ではない。")
    print("  ランダム化なしの世界で揃っていないのは Z だけだが、それだけで推定は壊れる。")

    print(f"\n  被覆 {res_conf.rate:.4f} は、95% と名乗る区間が {TRIALS:,} 回中 "
          f"{int((~res_conf.covered).sum()):,} 回も真値を外したということ。")
    print("  区間が短いことと、正しい場所にあることは別の話である。")

    draw(res_rand, res_conf)


if __name__ == "__main__":
    main()
