"""分岐した庭。24通りの分析経路を全部試すのと、1本を先に宣言しておくのとの差。

p ハッキングは「何度も検定して都合のよい1つを選ぶ」意図的な行為だと思われがちだが、
Gelman と Loken の言う *garden of forking paths* はもっと静かである。分析者は1回しか
検定していない。ただ、データを見てから「外れ値はどうするか」「共変量を入れるか」
「どの指標か」「どのセグメントか」を決めた。もし別のデータが出ていたら別の道を選んだ
——その**選ばなかった道の数**が、実質的な多重比較になる。

真の差がゼロのログに対して、外れ値2通り × 共変量2通り × 指標3本 × サブグループ2通り
= 24 経路を全部試して最小 p を採る手続きと、事前登録した1経路だけを回す手続きを
10,000 回比べる。

    uv run python examples/ch09/forking_paths_vs_preregistration.py
"""

import itertools

import numpy as np
from scipy import special

from toukei_tashikame import plots, sim

N = 40                 # 群あたりの人数
ALPHA = 0.05
TRIALS = 10_000
SEED = 96
Z_CUT = 2.5            # 「外れ値を落とす」ときの基準

OUTLIER = ("そのまま", "|z|>2.5 を除去")
COVARIATE = ("共変量なし", "年齢で調整")
METRIC = ("指標1", "指標2", "指標3")
SUBGROUP = ("全体", "新規ユーザーのみ")

# 経路の並び順。先頭が事前登録した1本（そのまま・共変量なし・指標1・全体）。
PATHS = list(itertools.product(range(2), range(2), range(3), range(2)))


def _t_and_df(y: np.ndarray, g: np.ndarray, age: np.ndarray, adjust: bool):
    """処置効果の t 統計量と自由度。``adjust=True`` なら年齢を共変量に入れる。

    共変量ありは $y = b_0 + b_1 g + b_2 \\text{age}$ の OLS で、$b_1$ の t を見る。
    共変量なしは Welch の t 検定。どちらも「1回の分析」として通用する。
    """
    n1, n2 = int((g == 0).sum()), int((g == 1).sum())
    if n1 < 4 or n2 < 4:
        return 0.0, 1.0
    if not adjust:
        a, b = y[g == 0], y[g == 1]
        s1, s2 = a.var(ddof=1) / n1, b.var(ddof=1) / n2
        if s1 + s2 <= 0.0:
            return 0.0, 1.0
        t = (b.mean() - a.mean()) / np.sqrt(s1 + s2)
        df = (s1 + s2) ** 2 / (s1**2 / (n1 - 1) + s2**2 / (n2 - 1))
        return float(t), float(df)

    x = np.column_stack([np.ones(y.size), g, age])
    xtx_inv = np.linalg.inv(x.T @ x)
    beta = xtx_inv @ (x.T @ y)
    resid = y - x @ beta
    dof = y.size - 3
    s2 = float(resid @ resid) / dof
    se = np.sqrt(s2 * xtx_inv[1, 1])
    return float(beta[1] / se), float(dof)


def one_trial(rng) -> np.ndarray:
    """真の差がゼロのログを1本作り、24 経路それぞれの p 値を返す。"""
    g = np.repeat([0, 1], N)
    age = rng.normal(0.0, 1.0, size=2 * N)          # 結果とは無関係な共変量
    is_new = rng.integers(0, 2, size=2 * N) == 1    # 事後に切りたくなるセグメント
    metrics = rng.normal(0.0, 1.0, size=(3, 2 * N))  # 真の効果はどれもゼロ

    stats_t = np.empty(len(PATHS))
    dfs = np.empty(len(PATHS))
    everyone = np.ones(2 * N, dtype=bool)
    # 標本の切り出しは（外れ値 × 指標 × サブグループ）の12通りしかない。共変量の有無は
    # 同じ標本の上での2通りなので、切り出しを共有して2回ぶんの分析を回す。
    for out_i in range(2):
        for met_i in range(3):
            for sub_i in range(2):
                keep = is_new if sub_i else everyone
                y = metrics[met_i]
                if out_i:
                    z = np.abs(y - y[keep].mean()) / y[keep].std(ddof=1)
                    keep = keep & (z <= Z_CUT)
                ys, gs, ages = y[keep], g[keep], age[keep]
                for cov_i in range(2):
                    k = PATHS.index((out_i, cov_i, met_i, sub_i))
                    stats_t[k], dfs[k] = _t_and_df(ys, gs, ages, bool(cov_i))

    return 2 * special.stdtr(dfs, -np.abs(stats_t))


def draw(p: np.ndarray, curve: np.ndarray) -> None:
    fig, axes = plots.figure(1, 2, w=2.0, h=0.95)

    ax = axes[0]
    ks = np.arange(1, len(PATHS) + 1)
    ax.plot(ks, curve, color=plots.PALETTE["reject"], lw=1.3, zorder=4)
    ax.fill_between(ks, ALPHA, curve, color=plots.PALETTE["reject"], alpha=0.20, lw=0)
    plots.mark_truth(ax, ALPHA, "事前登録した1経路 = 0.05", axis="y")
    ax.set_xlabel("試した経路の数")
    ax.set_ylabel("どれかが有意になる割合")
    ax.set_ylim(0, max(curve) * 1.15)
    ax.set_title("道が増えるほど、必ず何かが見つかる")

    ax = axes[1]
    best = p.min(axis=1)
    bins = np.linspace(0, 1, 41)
    ax.hist(p[:, 0], bins=bins, density=True, color=plots.PALETTE["data"], alpha=0.5,
            lw=0, label="事前登録の1経路")
    ax.hist(best, bins=bins, density=True, color=plots.PALETTE["reject"], alpha=0.55,
            lw=0, label="24経路の最小 p")
    ax.axvline(ALPHA, color=plots.PALETTE["reject"], lw=0.9, ls="--", dashes=(4, 2.2))
    ax.set_xlabel("p 値")
    ax.set_ylabel("密度")
    ax.legend(loc="upper right")
    ax.set_title("最小 p は一様分布ではなくなる")

    plots.save(fig, "fig-9-6-forking-paths.png")


def main() -> None:
    plots.setup()
    with sim.Timer("9-6 分岐した庭"):
        p = sim.repeat(one_trial, trials=TRIALS, seed=SEED, progress=False)

    prereg = float((p[:, 0] < ALPHA).mean())
    forking = float((p.min(axis=1) < ALPHA).mean())
    curve = np.array([(p[:, :k] < ALPHA).any(axis=1).mean() for k in range(1, len(PATHS) + 1)])
    per_path = (p < ALPHA).mean(axis=0)

    print(f"真の差はゼロ。n={N}/群、α={ALPHA}、{TRIALS:,} 回")
    print(f"経路 = 外れ値{len(OUTLIER)}通り × 共変量{len(COVARIATE)}通り × "
          f"指標{len(METRIC)}本 × サブグループ{len(SUBGROUP)}通り = {len(PATHS)} 通り\n")

    se_pre = np.sqrt(prereg * (1 - prereg) / TRIALS)
    se_fork = np.sqrt(forking * (1 - forking) / TRIALS)
    print(f"  事前登録した1経路だけを回す   {prereg:.4f} ± {1.96 * se_pre:.4f}")
    print(f"  24経路のどれかが有意になる    {forking:.4f} ± {1.96 * se_fork:.4f}"
          f"   （{forking / prereg:.1f} 倍）\n")

    print("経路ごとの偽陽性率（単独で見ればどれも 5% 前後で、何も間違っていない）:")
    order = np.argsort(-per_path)[:4]
    for k in order:
        out_i, cov_i, met_i, sub_i = PATHS[k]
        name = f"{OUTLIER[out_i]} / {COVARIATE[cov_i]} / {METRIC[met_i]} / {SUBGROUP[sub_i]}"
        print(f"  {name:<44}{per_path[k]:.4f}")
    print(f"  … 24経路の平均 {per_path.mean():.4f}\n")

    print("分析者は1回しか検定していない。それでも 24 本の道が存在したという事実だけで、")
    print("報告される p 値は名目の意味を失う。事前登録が効くのは、道を1本に減らすからである。")
    draw(p, curve)


if __name__ == "__main__":
    main()
