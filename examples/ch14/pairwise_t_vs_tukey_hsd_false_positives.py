"""差が無い5群を10,000回作り、「1つでも有意」がどれだけ出るかを3つの手続きで数える。

5群あればペアは10通りある。どのペアも真の差は0なのに、10回のt検定をそれぞれ α=0.05 で
やると、族全体で1回でも間違える確率は 0.05 では済まなくなる。10回が独立なら
$1-0.95^{10}=0.4013$ で、これが誤警報の確率の**上限**になる。

数え上げると実測は 0.29 前後で、上限より小さい。10個のペアは同じ5群を共有している
（群1対群2 と 群1対群3 は群1を分け合う）ので、検定どうしが正に相関し、当たり外れが
そろって動くからである。それでも名目の 0.05 の6倍で、「有意差が出たペアだけ報告する」が
なぜ再現しないかの説明には十分足りる。

分散分析はこれを1つの F 検定に畳んで α を守る。Tukey の HSD は10ペアを全部見たうえで、
帰無分布を studentized range に取り替えることで族全体の誤り率を α に抑える。数え上げると、
守れているものと守れていないものがはっきり分かれる。

速度のために、3つの手続きはどれも臨界値をあらかじめ1回だけ計算し、各試行では統計量と
比べるだけにしてある（p 値を10,000回計算するのと結論は同じ）。

    uv run python examples/ch14/pairwise_t_vs_tukey_hsd_false_positives.py
"""

import itertools

import numpy as np
from scipy import stats

from toukei_tashikame import plots, sim

K, N, SEED, TRIALS, ALPHA = 5, 20, 141, 10_000, 0.05
DF_W = K * (N - 1)                      # 群内自由度 95
PAIRS = list(itertools.combinations(range(K), 2))   # 10 ペア

# 臨界値は3つとも定数なので、ループの外で1度だけ引く。
T_CRIT = stats.t.ppf(1 - ALPHA / 2, df=2 * (N - 1))          # 各ペアは2群だけで組む
F_CRIT = stats.f.ppf(1 - ALPHA, K - 1, DF_W)
Q_CRIT = stats.studentized_range.ppf(1 - ALPHA, K, DF_W)


def one_trial(rng: np.random.Generator) -> tuple[bool, bool, bool]:
    """差が無い5群を作り、3つの手続きが「1つでも有意」と言うかを返す。"""
    y = rng.normal(0.0, 1.0, size=(K, N))       # 真の群平均はすべて 0
    m = y.mean(axis=1)
    v = y.var(axis=1, ddof=1)

    # (1) 総当たりのt検定。ペアごとに、そのペアの2群だけでプールした分散を使う。
    t = np.array([(m[i] - m[j]) / np.sqrt((v[i] + v[j]) / N) for i, j in PAIRS])
    any_t = bool((np.abs(t) > T_CRIT).any())

    # (2) 一元配置分散分析。10ペアを1つの F に畳む。
    ss_b = N * ((m - m.mean()) ** 2).sum()
    ss_w = ((N - 1) * v).sum()
    f = (ss_b / (K - 1)) / (ss_w / DF_W)
    any_f = bool(f > F_CRIT)

    # (3) Tukey HSD。全群まとめた MSE を使い、帰無分布を studentized range に取り替える。
    mse = ss_w / DF_W
    q = np.array([abs(m[i] - m[j]) / np.sqrt(mse / N) for i, j in PAIRS])
    any_q = bool((q > Q_CRIT).any())

    return any_t, any_f, any_q


def main() -> None:
    plots.setup()
    theory = 1.0 - (1.0 - ALPHA) ** len(PAIRS)

    with sim.Timer("14-1 の 10,000 回"):
        out = sim.repeat(one_trial, trials=TRIALS, seed=SEED, progress=False)
    rates = out.mean(axis=0)
    ses = np.sqrt(rates * (1 - rates) / TRIALS)

    print(f"--- 14-1 / 14-4 差の無い {K} 群（各 n={N}, seed={SEED}）を {TRIALS:,} 回 ---")
    print(f"  ペアは {len(PAIRS)} 通り。どのペアも真の差は 0。α={ALPHA}\n")
    print("  手続き                       1つでも有意になった割合    ±1.96SE")
    labels = ["総当たりのt検定（10回）", "一元配置分散分析（F 1回）", "Tukey HSD（10ペア同時）"]
    for lab, r, se in zip(labels, rates, ses, strict=True):
        print(f"  {lab:<26} {r:.4f}              ±{1.96 * se:.4f}")

    print(f"\n  10回が独立なら 1 - 0.95^{len(PAIRS)} = {theory:.4f}。これは上限で、"
          f"実測はそれより {theory - rates[0]:.4f} 小さい")
    print(f"  10ペアは同じ5群を共有していて正に相関するため、上限には届かない。"
          f"それでも名目の {rates[0] / ALPHA:.1f} 倍")
    print(f"  ANOVA と Tukey はどちらも名目の {ALPHA} を守っている。"
          "守り方が違うだけで、目標は同じ")
    print("\n  誤解しやすい点: ANOVA が有意でも「どのペアが違うか」は言えない。"
          "そこで総当たりのt検定に")
    print("  戻ると 0.29 の世界に逆戻りする。事後比較に Tukey を使うのはそのため")

    # --- 図 ---
    fig, ax = plots.figure(w=1.15)
    x = np.arange(3)
    ax.bar(x, rates, width=0.55, color=[plots.PALETTE["reject"], plots.PALETTE["estimate"],
                                        plots.PALETTE["estimate"]], lw=0, zorder=3)
    ax.errorbar(x, rates, yerr=1.96 * ses, fmt="none", ecolor=plots.PALETTE["ink"],
                elinewidth=0.8, capsize=2, zorder=4)
    plots.mark_truth(ax, theory, f"独立と仮定した上限 1 - 0.95^{len(PAIRS)} = {theory:.4f}",
                     axis="y")
    ax.axhline(ALPHA, color=plots.PALETTE["ink2"], lw=0.9, ls="--", dashes=(4, 2.2), zorder=4)
    ax.annotate(f"名目 α = {ALPHA}", xy=(0.02, ALPHA), xycoords=("axes fraction", "data"),
                ha="left", va="bottom", fontsize=6.0, color=plots.PALETTE["ink2"])
    for xi, r in zip(x, rates, strict=True):
        ax.annotate(f"{r:.4f}", xy=(xi, r), ha="center", va="bottom", fontsize=6.2,
                    xytext=(0, 3), textcoords="offset points")
    ax.set_xticks(x)
    ax.set_xticklabels(["総当たりt検定", "分散分析", "Tukey HSD"])
    ax.set_ylabel("1つでも有意になる割合")
    ax.set_ylim(0, 0.48)
    ax.set_title(f"差の無い5群、{TRIALS:,} 回")
    fig.tight_layout()
    plots.save(fig, "fig-14-1-pairwise-vs-tukey.png")


if __name__ == "__main__":
    main()
