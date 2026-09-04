"""A/Bテストの1つのログに、比率の検定を4通り当てて突き合わせる。

同じ 2×2 の表に、二項検定・比率差の z 検定・カイ二乗検定・Fisher の正確検定を当てる。
どれも「2つの比率は等しい」を検定しているのに、p 値は一致しない。違いは近似の仕方と、
何を固定して数えているかにある。

- z 検定と カイ二乗（補正なし）は**同じもの**である（z² = χ²）。数字が完全に一致する
- Yates の連続修正は、離散な度数を連続分布で近似する補正。保守側に寄る
- Fisher は周辺度数を固定して超幾何分布で数え上げる。近似ではないが、条件付きである

α=0.05 のすぐ近くにあるとき、この差が「有意/非有意」を分ける。どれを使うか**先に決めて
おく**しかない、という話の実例になる。

    uv run python examples/ch08/proportion_tests_for_ab.py
"""

import numpy as np
from scipy import stats

from toukei_tashikame import datasets, plots, testing

ALPHA = 0.05
N_A = N_B = 4000
P_A = 0.030
LOGS = 2_000   # 判定が割れる頻度を数えるためのログ本数


def pvalues_for(k_a, k_b):
    """度数の配列から、3通りの p 値をまとめて出す。χ² の 2×2 は閉じた式で書ける。"""
    a, b = k_b, N_B - k_b   # 行1: B
    c, d = k_a, N_A - k_a   # 行2: A
    n = N_A + N_B
    r1, r2 = a + b, c + d
    c1, c2 = a + c, b + d
    chi2_plain = n * (a * d - b * c) ** 2 / (r1 * r2 * c1 * c2)
    yates = n * np.maximum(np.abs(a * d - b * c) - n / 2, 0) ** 2 / (r1 * r2 * c1 * c2)
    fisher = np.array([stats.fisher_exact([[bb, nb], [aa, na]]).pvalue
                       for bb, nb, aa, na in zip(a, b, c, d, strict=True)])
    return stats.chi2.sf(chi2_plain, 1), stats.chi2.sf(yates, 1), fisher


def main() -> None:
    plots.setup()

    d = datasets.ab_test(n_a=4000, n_b=4000, p_a=0.030, lift=0.20, seed=87)
    k_a, n_a = int(d.a.sum()), d.a.size
    k_b, n_b = int(d.b.sum()), d.b.size

    print("--- ログ ---")
    print(f"  A: {k_a:4d} / {n_a}  = {k_a / n_a:.4f}   （真の CVR {d.p_a:.4f}）")
    print(f"  B: {k_b:4d} / {n_b}  = {k_b / n_b:.4f}   （真の CVR {d.p_b:.4f}）")
    print(f"  観測リフト = {(k_b / n_b) / (k_a / n_a) - 1:+.4f}   "
          f"（真のリフト {d.lift:+.4f}）")

    table = [[k_b, n_b - k_b], [k_a, n_a - k_a]]
    print(f"\n  2×2 の表（行=B,A / 列=CV,非CV）: {table}")

    results = [
        ("二項検定（B を p_A の下で）",
         testing.binom_test(k_b, n_b, p=k_a / n_a)),
        ("比率差の z 検定（プール）",
         testing.prop_2samp(k_b, n_b, k_a, n_a, method="score")),
        ("比率差の z 検定（Wald）",
         testing.prop_2samp(k_b, n_b, k_a, n_a, method="wald")),
        ("χ² 検定（補正なし）", testing.chi2_independence(table, correction=False)),
        ("χ² 検定（Yates 補正）", testing.chi2_independence(table, correction=True)),
        ("Fisher の正確検定", testing.fisher_exact(table)),
    ]

    print(f"\n--- 同じ表に当てた6通り（α={ALPHA}）---")
    print(f"{'検定':<28}{'統計量':>12}{'p 値':>12}   判定")
    for label, res in results:
        mark = "棄却" if res.pvalue < ALPHA else "棄却しない"
        print(f"{label:<30}{res.stat:>11.4f}{res.pvalue:>12.4f}   {mark}")

    z = results[1][1]
    chi2 = results[3][1]
    print(f"\n  z 検定（プール）の z² = {z.stat**2:.6f} と χ²（補正なし）"
          f"= {chi2.stat:.6f} は同じ数。差 = {abs(z.stat**2 - chi2.stat):.2e}")
    print("  「z 検定とカイ二乗検定のどちらを使うか」という問いには意味がない。同じ検定である")
    ps = [r.pvalue for _, r in results]
    print(f"  p 値の幅は {min(ps):.4f} 〜 {max(ps):.4f}。"
          "このログでは6通りのどれを使っても結論は同じ（棄却しない）")
    print(f"  真のリフトは {d.lift:+.2%} あるのに検出できていない。"
          "各群4,000件・CVR3%では、これは驚くことではない（第10章の検出力の話）")

    print("\n--- 仮定（p 値だけ取り出して捨てないために）---")
    print(f"  {results[3][1]}")

    # --- 検定の選び方で判定が割れる頻度を数える ---
    print(f"\n--- 同じ設計のログを {LOGS:,} 本作って、3通りの判定を比べる ---")
    rng = np.random.default_rng(870)
    scenarios = []
    for label, lift in (("帰無が真（lift=0）", 0.0), ("真のリフト +20%", 0.20)):
        ka = rng.binomial(N_A, P_A, size=LOGS)
        kb = rng.binomial(N_B, P_A * (1 + lift), size=LOGS)
        p_plain, p_yates, p_fisher = pvalues_for(ka, kb)
        rej = [(p_plain < ALPHA), (p_yates < ALPHA), (p_fisher < ALPHA)]
        disagree = float((rej[0] != rej[2]).mean())
        scenarios.append((label, [float(r.mean()) for r in rej], disagree))
        kind = "第一種の誤り" if lift == 0 else "検出力"
        print(f"  {label:<20} {kind}: χ²補正なし {rej[0].mean():.4f} / "
              f"Yates {rej[1].mean():.4f} / Fisher {rej[2].mean():.4f}")
        print(f"  {'':<20} χ²（補正なし）と Fisher で判定が割れたログ: {disagree:.4f}")
    print("\n  Yates と Fisher は保守側に寄る。α を守りたいなら安全だが、"
          "そのぶん検出力を捨てている")
    print("  どれを使うかは、データを見る**前**に決めておくこと（第9章）")

    # --- 図 ---
    fig, (ax1, ax2) = plots.figure(1, 2, w=1.7)

    # Wilson の区間。0 に近い比率では Wald より素直に振る舞う
    for i, (k, n, name) in enumerate([(k_a, n_a, "A"), (k_b, n_b, "B")]):
        ph, zc = k / n, 1.96
        center = (ph + zc**2 / (2 * n)) / (1 + zc**2 / n)
        half = zc / (1 + zc**2 / n) * np.sqrt(ph * (1 - ph) / n + zc**2 / (4 * n**2))
        ax1.plot([center - half, center + half], [i, i], lw=2.0,
                 color=plots.PALETTE["interval"], solid_capstyle="butt")
        ax1.plot([ph], [i], "o", ms=4, color=plots.PALETTE["estimate"])
        ax1.annotate(f"{name}  {ph:.4f}", xy=(ph, i), xytext=(0, 6),
                     textcoords="offset points", ha="center", fontsize=6.2)
    ax1.axvline(d.p_a, color=plots.PALETTE["truth"], lw=1.0, ls="--", dashes=(4, 2))
    ax1.axvline(d.p_b, color=plots.PALETTE["truth"], lw=1.0, ls="--", dashes=(4, 2))
    ax1.annotate("真の CVR", xy=(d.p_b, 1.35), fontsize=6.0,
                 color=plots.PALETTE["truth"], ha="left", xytext=(3, 0),
                 textcoords="offset points")
    ax1.set_ylim(-0.6, 1.6)
    ax1.set_yticks([])
    ax1.set_xlabel("CVR")
    ax1.set_title(f"観測 CVR と 95% 区間（各群 {n_a:,}）")

    labels = ["二項検定", "z 検定（プール）", "z 検定（Wald）", "χ²（補正なし）",
              "χ²（Yates）", "Fisher"]
    ys = np.arange(len(results))[::-1]
    colors = [plots.PALETTE["reject"] if p < ALPHA else plots.PALETTE["data"] for p in ps]
    ax2.barh(ys, ps, height=0.55, color=colors, alpha=0.85, lw=0)
    for y, p_val in zip(ys, ps, strict=True):
        ax2.annotate(f"{p_val:.4f}", xy=(p_val, y), xytext=(3, 0),
                     textcoords="offset points", va="center", fontsize=6.2)
    plots.mark_truth(ax2, ALPHA, "α = 0.05")
    ax2.set_yticks(ys)
    ax2.set_yticklabels(labels, fontsize=5.8)
    ax2.set_xlim(0, max(ps) * 1.35)
    ax2.set_xlabel("p 値")
    ax2.set_title("同じ表・同じ帰無仮説、違う p 値")

    fig.tight_layout()
    plots.save(fig, "fig-8-7-proportion-tests.png")


if __name__ == "__main__":
    main()
