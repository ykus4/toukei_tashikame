"""同じ人を5回測ったデータを「100人ぶん」として扱うと、第一種の誤りが跳ね上がる。

被験者20人を4群に分け、1人につき5回測る。行数は 100 になるが、独立な情報は 100 個も
ない。同じ人の5回は互いに似ている（級内相関 ICC=0.6）からで、それでも通常の一元配置
分散分析に 100 行をそのまま流し込むと、誤差分散が小さく見積もられ、F が大きく出る。

正しくは被験者を変量効果に入れる。釣り合い型（各人が同じ回数）のときは、これは
「1人を1つの値（5回の平均）に畳んでから群を比べる」のとまったく同じ F になるので、
10,000 回まわす側は畳んだ形で計算し、混合効果モデルとの一致は1本のデータで確かめる。

有効サンプルサイズは n / (1 + (m-1)·ICC) に目減りする。100 行が実質いくつぶんなのかを
先に見ておくと、「行が増えたのに検定が強くならない」という感触が数字になる。

    uv run python examples/ch14/repeated_measures_as_independent_inflates_error.py
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

from toukei_tashikame import plots, sim

N_GROUP, N_SUBJ_PER_GROUP, N_REP = 4, 5, 5      # 4群 × 5人 × 5回 = 100 行
N_SUBJ = N_GROUP * N_SUBJ_PER_GROUP
ICC, SEED, TRIALS, ALPHA = 0.6, 146, 10_000, 0.05
GROUP_OF_SUBJ = np.repeat(np.arange(N_GROUP), N_SUBJ_PER_GROUP)


def make_data(rng: np.random.Generator, icc: float) -> np.ndarray:
    """被験者効果 + 測定誤差。真の群差はゼロ。``datasets.clustered`` と同じ作り。"""
    subject = rng.normal(0.0, np.sqrt(icc), size=N_SUBJ)            # 個人差
    within = rng.normal(0.0, np.sqrt(1.0 - icc), size=(N_SUBJ, N_REP))
    return subject[:, None] + within                                # (20, 5)


def f_pvalue(groups: list[np.ndarray]) -> float:
    """釣り合い型の一元配置 F 検定。平方和の分解から直に組む。"""
    allv = np.concatenate(groups)
    grand = allv.mean()
    k, n = len(groups), allv.size
    ss_b = sum(a.size * (a.mean() - grand) ** 2 for a in groups)
    ss_w = sum(((a - a.mean()) ** 2).sum() for a in groups)
    f = (ss_b / (k - 1)) / (ss_w / (n - k))
    return float(stats.f.sf(f, k - 1, n - k))


def one_trial(rng: np.random.Generator, icc: float = ICC) -> tuple[float, float]:
    """(独立扱いの p, 被験者を変量効果に入れた p) を返す。真の群差は 0。"""
    y = make_data(rng, icc)
    naive = f_pvalue([y[GROUP_OF_SUBJ == k].ravel() for k in range(N_GROUP)])   # 100 行
    means = y.mean(axis=1)                                                     # 1人1値
    correct = f_pvalue([means[GROUP_OF_SUBJ == k] for k in range(N_GROUP)])    # 20 行
    return naive, correct


def show_mixed_model_agreement() -> None:
    """畳んだ F と混合効果モデルが同じものを見ていることを、1本のデータで確かめる。"""
    y = make_data(np.random.default_rng(SEED), ICC)
    df = pd.DataFrame({
        "y": y.ravel(),
        "group": np.repeat(GROUP_OF_SUBJ, N_REP),
        "subject": np.repeat(np.arange(N_SUBJ), N_REP),
    })
    fit = smf.mixedlm("y ~ C(group)", df, groups=df["subject"]).fit()
    wald = fit.wald_test_terms(scalar=False).table.loc["C(group)"]
    chi2, df_c = float(np.ravel(wald["statistic"])[0]), int(wald["df_constraint"])

    means = y.mean(axis=1)
    f_fold = stats.f_oneway(*[means[GROUP_OF_SUBJ == k] for k in range(N_GROUP)])
    var_subj, var_err = float(fit.cov_re.iloc[0, 0]), float(fit.scale)

    print("\n  混合効果モデル（被験者を変量効果に）を1本のデータで当てると")
    print(f"    被験者の分散 {var_subj:.4f} / 残差の分散 {var_err:.4f}"
          f"  → ICC の推定 {var_subj / (var_subj + var_err):.4f}（真値 {ICC}）")
    print(f"    群の Wald χ² = {chi2:.4f}（df {df_c}）、χ²/df = {chi2 / df_c:.4f}")
    print(f"    5回を平均してからの F = {f_fold.statistic:.4f}"
          f"   （差 {abs(chi2 / df_c - f_fold.statistic):.2e}）")
    print("    ← 釣り合い型なら同じ量。χ² は自由度を無限とみなす近似なので、"
          "p はわずかに小さく出る")


def main() -> None:
    plots.setup()
    deff = 1 + (N_REP - 1) * ICC
    n_row = N_SUBJ * N_REP

    with sim.Timer("14-6 の 10,000 回"):
        p = sim.repeat(one_trial, trials=TRIALS, seed=SEED, progress=False)
    rate = (p < ALPHA).mean(axis=0)
    se = np.sqrt(rate * (1 - rate) / TRIALS)

    print(f"--- 14-6 被験者{N_SUBJ}人 × {N_REP}回 = {n_row}行、ICC={ICC}、"
          f"真の群差 0 を {TRIALS:,} 回 ---")
    print(f"  {N_GROUP}群（1群あたり {N_SUBJ_PER_GROUP} 人）。α={ALPHA}\n")
    print("  扱い方                                第一種の誤り     ±1.96SE")
    print(f"  {n_row}行を独立扱い（通常のANOVA）          {rate[0]:.4f}          ±{1.96 * se[0]:.4f}")
    print(f"  被験者を変量効果に（{N_SUBJ}人で比べる）      {rate[1]:.4f}          ±{1.96 * se[1]:.4f}")
    print(f"\n  独立扱いは名目の {rate[0] / ALPHA:.1f} 倍。"
          "「行が増えれば検出力が上がる」は独立なときだけの話")
    print(f"  デザイン効果 1 + ({N_REP}-1)×{ICC} = {deff:.2f}")
    print(f"  有効サンプルサイズ {n_row} / {deff:.2f} = {n_row / deff:.1f} 行ぶん"
          f"（被験者{N_SUBJ}人という上限には届かない）")

    show_mixed_model_agreement()

    # --- ICC を動かすと、独立扱いの誤り率がどう伸びるか ---
    grid = [0.0, 0.2, 0.4, 0.6, 0.8]
    sweep_trials = 2_000
    naive_rates, correct_rates = [], []
    for icc in grid:
        pv = sim.repeat(lambda rng, i=icc: one_trial(rng, i), trials=sweep_trials,
                        seed=SEED + 1, progress=False)
        r = (pv < ALPHA).mean(axis=0)
        naive_rates.append(r[0])
        correct_rates.append(r[1])

    print(f"\n  ICC を動かす（各 {sweep_trials:,} 回）")
    print("    ICC     独立扱い   変量効果   デザイン効果   有効n")
    for icc, a, b in zip(grid, naive_rates, correct_rates, strict=True):
        d = 1 + (N_REP - 1) * icc
        print(f"    {icc:.1f}     {a:.4f}     {b:.4f}       {d:.2f}       {n_row / d:5.1f}")
    print("    ← ICC=0（本当に独立）なら両者は一致する。"
          "問題は相関そのものではなく、相関を無いことにする点にある")

    # --- 図 ---
    fig, axes = plots.figure(1, 2, w=1.9, h=1.0)

    ax = axes[0]
    ax.plot(grid, naive_rates, color=plots.PALETTE["reject"], lw=1.4, marker="o", ms=3.5, zorder=4)
    ax.plot(grid, correct_rates, color=plots.PALETTE["estimate"], lw=1.4, marker="s", ms=3.5,
            zorder=4)
    ax.annotate("独立扱い", xy=(grid[-1], naive_rates[-1]), xytext=(-4, -8),
                textcoords="offset points", ha="right", va="top", fontsize=6.2,
                color=plots.PALETTE["reject"])
    ax.annotate("被験者を変量効果に", xy=(grid[1], correct_rates[1]), xytext=(0, 12),
                textcoords="offset points", ha="center", va="bottom", fontsize=6.2,
                color=plots.PALETTE["estimate"])
    plots.mark_truth(ax, ALPHA, f"名目 α = {ALPHA}", axis="y")
    ax.set_xlabel("級内相関 ICC")
    ax.set_ylabel("第一種の誤り")
    ax.set_ylim(0, max(naive_rates) * 1.18)
    ax.set_title(f"ICC が上がるほど独立扱いは壊れる（各 {sweep_trials:,} 回）")

    ax = axes[1]
    bins = np.linspace(0, 1, 21)
    ax.hist(p[:, 0], bins=bins, density=True, color=plots.PALETTE["reject"], alpha=0.55, lw=0,
            zorder=2)
    ax.hist(p[:, 1], bins=bins, density=True, histtype="step",
            color=plots.PALETTE["estimate"], lw=1.3, zorder=4)
    ax.axhline(1.0, color=plots.PALETTE["truth"], lw=1.1, ls="--", dashes=(4, 2.0), zorder=5)
    ax.annotate("帰無仮説が正しいときの p は一様", xy=(0.5, 1.0), xytext=(0, 4),
                textcoords="offset points", ha="center", va="bottom", fontsize=6.0,
                color=plots.PALETTE["truth"])
    ax.annotate("独立扱い（0 に寄る）", xy=(0.06, 2.2), fontsize=6.2,
                color=plots.PALETTE["reject"], ha="left", va="bottom")
    ax.set_xlabel("p 値")
    ax.set_ylabel("密度")
    ax.set_title(f"ICC={ICC} での p 値の分布")
    fig.tight_layout()
    plots.save(fig, "fig-14-6-icc-inflates-alpha.png")


if __name__ == "__main__":
    main()
