"""比率の区間を3つ並べる — Wald は名目に届かず、Wilson は当たり、Clopper-Pearson は広い。

n=20, p=0.1 という「実務でよくある小標本・低頻度」の場面で、3つのレシピの被覆を
10,000 回で数え上げる。さらに p を 0.01〜0.50 で掃いて被覆曲線を描く。n=20 なら
起こりうる結果は k=0,1,…,20 の21通りしかないので、曲線のほうは二項確率で重みを
つけて厳密に足し上げられる。数え上げと厳密計算が一致することも同時に確かめる。

Wald の壊れ方は数字より先に形に出る。k=0 のとき区間は [0, 0]、幅ゼロである。
「95% の確率で真の比率はちょうど 0 です」と言っていることになる。

    uv run python examples/ch06/proportion_ci_coverage_wald_wilson_cp.py
"""

import numpy as np
from scipy import stats

from toukei_tashikame import estimate, plots, sim

N, P_TRUE, CONF = 20, 0.10, 0.95
TRIALS, SEED = 10_000, 29
METHODS = {
    "Wald": estimate.ci_prop_wald,
    "Wilson": estimate.ci_prop_wilson,
    "Clopper-Pearson": estimate.ci_prop_clopper_pearson,
}
# 役割で色を引く。壊れるほうをオレンジ（棄却域と同じ「注意」の色）にする。
ROLE = {"Wald": "reject", "Wilson": "estimate", "Clopper-Pearson": "data"}
STYLE = {"Wald": "--", "Wilson": "-", "Clopper-Pearson": ":"}


def exact_coverage(ci_fn, p: float, n: int = N, conf: float = CONF) -> float:
    """被覆確率を厳密に出す。起こりうる k は n+1 通りしかないので全部足せる。

    シミュレーションは「数え上げの誤差」を必ず伴うが、離散分布ならその誤差なしで
    同じ量を出せる。曲線のギザギザが乱数のせいではないことが、これで分かる。
    """
    k = np.arange(n + 1)
    iv = np.array([ci_fn(int(i), n, conf) for i in k])
    hit = (iv[:, 0] <= p) & (p <= iv[:, 1])
    return float(stats.binom.pmf(k, n, p)[hit].sum())


def main() -> None:
    plots.setup()

    # 同じ seed を使うので、3つのレシピは全く同じ 10,000 個の k を見ることになる。
    ks = sim.repeat(lambda rng: int(rng.binomial(N, P_TRUE)), trials=TRIALS, seed=SEED,
                    progress=False)

    print(f"--- n={N}, p={P_TRUE}（真値）で {TRIALS:,} 回。名目 {CONF:.0%} ---")
    print(f"  {'レシピ':<18}{'被覆':>10}{'誤差':>10}{'厳密値':>10}{'平均幅':>10}")
    results = {}
    for name, fn in METHODS.items():
        res = sim.coverage(lambda rng, f=fn: f(int(rng.binomial(N, P_TRUE)), N, CONF),
                           truth=P_TRUE, trials=TRIALS, seed=SEED, progress=False)
        width = float(np.mean(res.intervals[:, 1] - res.intervals[:, 0]))
        exact = exact_coverage(fn, P_TRUE)
        results[name] = (res, width, exact)
        print(f"  {name:<18}{res.rate:>10.4f}{1.96 * res.se:>+10.4f}{exact:>10.4f}{width:>10.4f}")
    print("  数え上げと厳密値の差は、どれもシミュレーション誤差の中に収まる")

    n_zero = int(np.sum(ks == 0))
    print("\n--- Wald が壊れる瞬間 ---")
    print(f"  k=0（20回中1回も起きなかった）が {n_zero:,} 回 / {TRIALS:,}"
          f"（理論 {(1 - P_TRUE) ** N:.4f} → {(1 - P_TRUE) ** N * TRIALS:.0f} 回）")
    for name, fn in METHODS.items():
        lo, hi = fn(0, N, CONF)
        print(f"  k=0 のときの区間  {name:<18}[{lo:.4f}, {hi:.4f}]   幅 {hi - lo:.4f}")
    print(f"  Wald はこの {n_zero:,} 回すべてで幅0の区間を返す。"
          "この時点で被覆の上限が失われている")
    below = sum(1 for name in METHODS if results[name][0].rate < CONF)
    print(f"  3つのうち名目を割ったのは {below} 個（Wald だけが大きく下回る）")

    print("\n--- p を 0.01〜0.50 で掃く（厳密計算）---")
    ps = np.linspace(0.01, 0.50, 99)
    curves = {name: np.array([exact_coverage(fn, p) for p in ps])
              for name, fn in METHODS.items()}
    print(f"  {'レシピ':<18}{'最小':>10}{'平均':>10}{'最大':>10}{'名目割れの割合':>16}")
    for name, c in curves.items():
        print(f"  {name:<18}{c.min():>10.4f}{c.mean():>10.4f}{c.max():>10.4f}"
              f"{np.mean(c < CONF):>16.2f}")
    for name, c in curves.items():
        print(f"  {name:<18}最小になる p = {ps[int(np.argmin(c))]:.2f}")
    print("  どの曲線もギザギザに揺れる。k が整数しか取れない以上、"
          "被覆を p によらず 0.95 ちょうどにすることはできない")
    print("  実務の既定は Wilson でよい。Clopper-Pearson は必ず 0.95 以上だが、"
          "そのぶん区間が広い（＝検出力を捨てている）")

    fig, axes = plots.figure(1, 2, w=2.0)

    ax = axes[0]
    # 凡例を置かず、曲線そのものの脇にラベルを書く（重ならない p を役割ごとに決める）
    label_at = {"Wald": (0.42, -9), "Wilson": (0.19, -10), "Clopper-Pearson": (0.24, 4)}
    for name, c in curves.items():
        line, = ax.plot(ps, c, color=plots.PALETTE[ROLE[name]], lw=1.1, ls=STYLE[name])
        if STYLE[name] == "--":
            line.set_dashes((4, 2.0))
        at, dy = label_at[name]
        i = int(np.argmin(np.abs(ps - at)))
        ax.annotate(name, xy=(ps[i], c[i]), xytext=(0, dy), textcoords="offset points",
                    ha="center", fontsize=6.0, color=plots.PALETTE[ROLE[name]])
    plots.mark_truth(ax, CONF, f"名目 {CONF:.0%}", axis="y")
    ax.annotate(f"Wald は p={ps[0]:.2f} で {curves['Wald'][0]:.2f} まで落ちる（枠の外）",
                xy=(0.03, 0.03), xycoords="axes fraction", fontsize=5.8,
                color=plots.PALETTE[ROLE["Wald"]])
    ax.set_xlabel("真の比率 p")
    ax.set_ylabel("被覆確率")
    ax.set_ylim(0.70, 1.02)
    ax.set_title(f"被覆曲線（n={N}, 厳密）")

    ax = axes[1]
    for k in range(7):
        for j, (name, fn) in enumerate(METHODS.items()):
            lo, hi = fn(k, N, CONF)
            y = k + 0.26 * (1 - j)
            ax.plot([lo, hi], [y, y], color=plots.PALETTE[ROLE[name]], lw=1.6,
                    solid_capstyle="butt", zorder=3)
            if hi - lo < 1e-9:   # 幅0は線にならないので点で示す
                ax.plot([lo], [y], "o", ms=2.6, color=plots.PALETTE[ROLE[name]], zorder=4)
            if k == 6:
                ax.annotate(name, xy=(hi, y), xytext=(3, 0), textcoords="offset points",
                            va="center", fontsize=5.8, color=plots.PALETTE[ROLE[name]])
    plots.mark_truth(ax, P_TRUE, f"真値 p = {P_TRUE}")
    ax.set_yticks(range(7))
    ax.set_yticklabels([f"k={k}" for k in range(7)])
    ax.set_xlim(-0.03, 0.85)
    ax.set_ylim(-0.6, 7.0)
    ax.set_xlabel("p の 95% 区間")
    ax.set_title("観測 k ごとの区間（k=0 で Wald は潰れる）")
    plots.save(fig, "fig-6-7-proportion-coverage.png")


if __name__ == "__main__":
    main()
