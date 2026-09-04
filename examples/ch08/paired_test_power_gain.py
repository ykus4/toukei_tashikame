"""対応を使うと検出力がどれだけ上がるか。前後データで数える。

個人差が大きいデータでは、群間の平均差は個人差のノイズに埋もれる。対応のある検定は
**同じ人の前後の差**を見るので、個人差が引き算で消える。消えるのは「効果」ではなく
「誰がもともと高いか」だけなので、失うものは何もない。

処置効果 1.0、個人差 σ_subj=5、測定ノイズ σ_within=1.0 の前後データ（n=30）を
10,000 組作り、対応のある t 検定と、対応を無視した2標本 t 検定の検出力を比べる。

    uv run python examples/ch08/paired_test_power_gain.py
"""

import numpy as np
from scipy import stats

from toukei_tashikame import plots

N, TRIALS, SEED, ALPHA = 30, 10_000, 84, 0.05
EFFECT, SD_SUBJ, SD_WITHIN = 1.0, 5.0, 1.0


def make(rng, trials: int):
    """``(trials, N)`` の前後データを一度に作る。被験者効果は前後で共有される。"""
    subj = rng.normal(0.0, SD_SUBJ, size=(trials, N))
    before = subj + rng.normal(0.0, SD_WITHIN, size=(trials, N))
    after = subj + EFFECT + rng.normal(0.0, SD_WITHIN, size=(trials, N))
    return before, after


def se(rate: float) -> float:
    return float(np.sqrt(rate * (1 - rate) / TRIALS))


def main() -> None:
    plots.setup()

    rng = np.random.default_rng(SEED)
    before, after = make(rng, TRIALS)

    # 対応あり: 差を取ってから1標本 t。個人差 σ_subj はここで消える
    p_paired = stats.ttest_rel(after, before, axis=1).pvalue
    # 対応なし: 同じデータを「別々の2群」と見なす。個人差が誤差に残る
    p_unpaired = stats.ttest_ind(after, before, axis=1, equal_var=False).pvalue

    pow_paired = float((p_paired < ALPHA).mean())
    pow_unpaired = float((p_unpaired < ALPHA).mean())

    # 参考: そもそも別々の人で測る設計（対応なし設計）。個人差は消せないが、
    # 2群が本当に独立なので標準誤差の見積もり自体は正しい
    rng2 = np.random.default_rng(SEED + 1)
    ctrl = rng2.normal(0.0, np.sqrt(SD_SUBJ**2 + SD_WITHIN**2), size=(TRIALS, N))
    trt = rng2.normal(EFFECT, np.sqrt(SD_SUBJ**2 + SD_WITHIN**2), size=(TRIALS, N))
    pow_between = float((stats.ttest_ind(trt, ctrl, axis=1, equal_var=False).pvalue < ALPHA).mean())

    print(f"処置効果 = {EFFECT}, 個人差 σ_subj = {SD_SUBJ}, 測定ノイズ σ_within = {SD_WITHIN}")
    print(f"n = {N}, {TRIALS:,}回, α = {ALPHA}\n")
    print(f"  対応のある t 検定    検出力 = {pow_paired:.4f} ± {1.96 * se(pow_paired):.4f}")
    hit_unpaired = int((p_unpaired < ALPHA).sum())
    print(f"  対応を無視した2標本  検出力 = {pow_unpaired:.4f}"
          f"（棄却 {hit_unpaired:,} / {TRIALS:,}回）   ← 同じデータを独立2群として扱った")
    print(f"  別々の人で測る設計   検出力 = {pow_between:.4f} ± {1.96 * se(pow_between):.4f}"
          "   ← 参考。そもそも対応を取らなかった場合")
    print(f"\n  同じデータ・同じ効果なのに {pow_paired:.4f} と {pow_unpaired:.4f}。"
          "違うのは検定の選び方だけである")
    print("  対応のあるデータを独立2標本にかけると、個人差のぶんまで誤差に数えてしまい、"
          "標準誤差が過大になる")

    # なぜ差がつくのか — 分母に入るばらつきを並べる
    sd_diff = float(np.sqrt(2) * SD_WITHIN)
    sd_group = float(np.sqrt(SD_SUBJ**2 + SD_WITHIN**2))
    print("\n--- 分母に入るばらつき（理論値）---")
    print(f"  対応あり: 差の標準偏差 sqrt(2)·σ_within        = {sd_diff:.4f}")
    print(f"  対応なし: 各群の標準偏差 sqrt(σ_subj²+σ_within²) = {sd_group:.4f}")
    print(f"  個人差 σ_subj={SD_SUBJ} は差分で消える。分母が {sd_group / sd_diff:.1f} 倍違えば、"
          "同じ効果の見えやすさもそれだけ変わる")
    print(f"  対応を無視した側が {TRIALS:,}回で一度も棄却しない（{hit_unpaired}回）のは"
          "丸め落ちではない。個人差が前後で共有されているぶん、")
    print(f"  平均差の分子はほぼ {EFFECT} に張りついて動かず、分母だけが"
          f"{sd_group / sd_diff:.1f}倍に膨らむ。|t| は 1 前後をうろつき、臨界値に届かない")
    print(f"\n  実測の相関 corr(before, after) = "
          f"{np.mean([np.corrcoef(b, a)[0, 1] for b, a in zip(before[:500], after[:500], strict=True)]):.4f}"
          "   ← この相関が高いほど対応の利得は大きい")

    # --- 図 ---
    fig, (ax1, ax2) = plots.figure(1, 2, w=1.5)

    b0, a0 = before[0], after[0]
    for bi, ai in zip(b0, a0, strict=True):
        ax1.plot([0, 1], [bi, ai], color=plots.PALETTE["data"], lw=0.6, alpha=0.6, zorder=2)
    ax1.plot([0, 1], [b0.mean(), a0.mean()], color=plots.PALETTE["estimate"], lw=1.8,
             marker="o", ms=3, zorder=5)
    ax1.annotate(f"平均 {a0.mean() - b0.mean():+.2f}", xy=(1, a0.mean()), xytext=(-2, 6),
                 textcoords="offset points", ha="right", fontsize=6.0,
                 color=plots.PALETTE["estimate"])
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(["前", "後"])
    ax1.set_xlim(-0.25, 1.25)
    ax1.set_ylabel("測定値")
    ax1.set_title(f"1組ぶんの前後（n={N}）— 線の上下動が個人差")

    values = [pow_paired, pow_unpaired, pow_between]
    bars = ax2.bar([0, 1, 2], values, width=0.5,
                   color=[plots.PALETTE["estimate"], plots.PALETTE["reject"],
                          plots.PALETTE["data"]],
                   alpha=0.85, lw=0)
    for rect, v in zip(bars, values, strict=True):
        ax2.annotate(f"{v:.4f}", xy=(rect.get_x() + rect.get_width() / 2, v),
                     xytext=(0, 2), textcoords="offset points", ha="center", fontsize=6.5)
    ax2.set_xticks([0, 1, 2])
    ax2.set_xticklabels(["対応あり", "対応を無視", "対応なし設計"])
    ax2.set_ylim(0, 1.08)
    ax2.set_ylabel("検出力")
    ax2.set_title(f"検出力（{TRIALS:,}回, α={ALPHA}）")

    fig.tight_layout()
    plots.save(fig, "fig-8-4-paired-vs-unpaired-power.png")


if __name__ == "__main__":
    main()
