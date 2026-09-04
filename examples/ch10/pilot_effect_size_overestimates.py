"""パイロットの効果量で本試験を設計すると、検出力は設計値に届かない。

「まず n=20 で予備実験をして、出た効果量でサンプルサイズを決めよう」という手順は自然に
見える。だが小さいパイロットの効果量は激しくばらつき、しかも**有意だったパイロットだけを
採用する**（有意でなければ本試験に進まない）ので、採用される効果量は勝者の呪いで系統的に
過大になる。過大な効果量からは過小な n が出る。

真の効果 d=0.3 に対して、この手順で設計した本試験の実際の検出力を数える。設計書には
0.80 と書いてあるのに、実際には半分以下しか出ない。

    uv run python examples/ch10/pilot_effect_size_overestimates.py
"""

from functools import cache

import numpy as np
from scipy import stats

from toukei_tashikame import plots, power

D_TRUE, N_PILOT, ALPHA, TARGET, TRIALS, SEED = 0.3, 20, 0.05, 0.80, 5_000, 107


def t_batch(n: int, d_true: float, trials: int, rng):
    """(trials, n) を2枚引いて t 検定をまとめてかけ、(p値, 観測効果量) を返す。"""
    a = rng.normal(0.0, 1.0, size=(trials, n))
    b = rng.normal(d_true, 1.0, size=(trials, n))
    sp = np.sqrt((a.var(axis=1, ddof=1) + b.var(axis=1, ddof=1)) / 2)
    d_hat = (b.mean(axis=1) - a.mean(axis=1)) / sp
    p = 2 * stats.t.sf(np.abs(d_hat) * np.sqrt(n / 2), 2 * (n - 1))
    return p, d_hat


@cache
def design_n(d_rounded: float) -> int:
    """効果量から本試験の n を決める（設計は小数2桁に丸めた効果量で行う）。"""
    return power.n_for_power(d_rounded, power=TARGET, alpha=ALPHA)


def main() -> None:
    plots.setup()
    rng = np.random.default_rng(SEED)

    # --- 1. パイロットを回す。有意だったものだけが本試験に進む ---
    p_all, d_all = t_batch(N_PILOT, D_TRUE, 20 * TRIALS, rng)
    sig = p_all < ALPHA
    d_pilot = d_all[sig][:TRIALS]          # 有意だったパイロットを TRIALS 本ぶん採用
    print(f"--- パイロット n={N_PILOT}/群、真の効果 d={D_TRUE} を {20 * TRIALS:,} 本 ---")
    print(f"  パイロットが有意になる割合（＝そもそもの検出力）{sig.mean():.4f}"
          f"（理論 {power.power_ttest(N_PILOT, D_TRUE, ALPHA):.4f}）")
    print(f"  全パイロットの効果量の平均            {d_all.mean():.4f}   ← 真値に当たる")
    print(f"  有意だったパイロットの効果量の平均    {d_pilot.mean():.4f}"
          f"   ← 真値の {d_pilot.mean() / D_TRUE:.1f} 倍")
    print(f"  （うち符号が逆のまま有意だったもの {int((d_pilot < 0).sum())} 本）")

    # --- 2. その効果量で本試験を設計し、真の効果 d=0.3 のもとで回す ---
    ns = np.array([design_n(round(abs(float(d)), 2)) for d in d_pilot])
    rejected = np.zeros(TRIALS, dtype=bool)
    for n in np.unique(ns):
        idx = np.flatnonzero(ns == n)
        p_conf, _ = t_batch(int(n), D_TRUE, idx.size, rng)
        rejected[idx] = p_conf < ALPHA
    achieved = float(rejected.mean())

    n_true = design_n(D_TRUE)
    p_ok, _ = t_batch(n_true, D_TRUE, TRIALS, rng)

    print(f"\n--- パイロットの効果量で設計した本試験を {TRIALS:,} 回 ---")
    print(f"  設計された n の中央値 {int(np.median(ns))} / 群"
          f"（最小 {ns.min()} 〜 最大 {ns.max()}）")
    print(f"  設計書に書かれた検出力  {TARGET:.4f}")
    print(f"  実際に出た検出力        {achieved:.4f}"
          f"   ← 設計値の {achieved / TARGET:.2f} 倍しかない")
    print(f"\n--- 参考: 真の効果 d={D_TRUE} を知って設計したら ---")
    print(f"  必要な n {n_true} / 群、実測検出力 {(p_ok < ALPHA).mean():.4f}")
    print(f"  パイロットは n を {n_true / np.median(ns):.1f} 分の1に見積もっていた。")
    print("  パイロットは「効果があるかどうか」の確認には使えるが、"
          "効果量の推定には使えない。")

    # --- 図 ---
    fig, axes = plots.figure(1, 2, w=1.7, h=1.0)
    ax = axes[0]
    bins = np.linspace(-1.5, 2.5, 70)
    ax.hist(d_all, bins=bins, color=plots.PALETTE["data"], alpha=0.45, lw=0,
            density=True)
    ax.hist(d_all[sig], bins=bins, color=plots.PALETTE["reject"], alpha=0.75, lw=0,
            density=True)
    plots.mark_truth(ax, D_TRUE, f"真値 {D_TRUE}")
    ax.axvline(d_pilot.mean(), color=plots.PALETTE["estimate"], lw=1.1, ls="--",
               dashes=(4, 2.0), zorder=6)
    ax.annotate(f"有意なパイロットの平均 {d_pilot.mean():.2f}", xy=(d_pilot.mean(), 0.80),
                xycoords=("data", "axes fraction"), xytext=(3, 0),
                textcoords="offset points", fontsize=6.0,
                color=plots.PALETTE["estimate"])
    ax.set_xlabel("パイロットの効果量 $\\hat{d}$")
    ax.set_ylabel("密度")
    ax.set_title(f"n={N_PILOT}/群のパイロット（灰＝全部、橙＝有意なものだけ再正規化）")

    ax = axes[1]
    labels = ["設計書の\n検出力", "パイロットで\n設計した実測", f"真値 d={D_TRUE} で\n設計した実測"]
    values = [TARGET, achieved, float((p_ok < ALPHA).mean())]
    colors = [plots.PALETTE["ink2"], plots.PALETTE["reject"], plots.PALETTE["estimate"]]
    ax.bar(labels, values, color=colors, width=0.55, zorder=3)
    for i, v in enumerate(values):
        ax.annotate(f"{v:.3f}", xy=(i, v), xytext=(0, 2), textcoords="offset points",
                    ha="center", fontsize=6.2, color=plots.PALETTE["ink"])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("検出力")
    ax.set_title("設計値と実際")
    fig.tight_layout()
    plots.save(fig, "fig-10-7-pilot-effect-size.png")


if __name__ == "__main__":
    main()
