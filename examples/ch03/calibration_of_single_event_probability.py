"""「明日の降水確率70%」は、その1日だけでは正しさを検証できない。1,000日なら数えられる。

明日は1回しか来ない。降ったか降らなかったかの2択しか残らないので、70% という宣言が
当たったのか外れたのかを、その日だけを見て決める方法はない。単一事象の確率は、頻度の
定義がそのままでは当てはまらないところである。

ではどうするか。**同じ確率を宣言した日を集めて数える。** 70% と言った日のうち本当に
70% で降っていれば、その予報はキャリブレートされている。ここでは 1,000 日ぶんの予報を
作り、宣言確率ごとの実現率を数え、Brier スコアで全体の当たり具合を1つの数にまとめる。

    uv run python examples/ch03/calibration_of_single_event_probability.py
"""

import numpy as np

from toukei_tashikame import plots

N_DAYS = 1_000
SEED = 11
LEVELS = np.round(np.arange(0.0, 1.01, 0.1), 1)


def forecast(rng: np.random.Generator, bias: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """1,000日ぶんの（宣言確率, 実際に降ったか）を作る。

    宣言確率は 0.0〜1.0 の 0.1 刻み。中央付近が多くなるように Beta(2,2) から引いて
    丸める（毎日 0% か 100% と言い切る予報士はいない）。``bias=1.0`` なら
    宣言どおりの確率で降る＝完全にキャリブレートされた予報士。``bias>1`` にすると
    宣言よりも実際の確率が下がる＝雨を言い過ぎる予報士になる。
    """
    declared = np.round(rng.beta(2.0, 2.0, size=N_DAYS), 1)
    true_p = np.clip(declared**bias, 0.0, 1.0)
    rained = rng.random(N_DAYS) < true_p
    return declared, rained


def calibration_table(declared: np.ndarray, rained: np.ndarray) -> list[tuple]:
    """宣言確率ごとに（宣言, 日数, 降った日数, 実現率）を並べる。"""
    rows = []
    for level in LEVELS:
        m = declared == level
        if m.sum() == 0:
            continue
        rows.append((float(level), int(m.sum()), int(rained[m].sum()), float(rained[m].mean())))
    return rows


def main() -> None:
    plots.setup()

    rng = np.random.default_rng(SEED)
    declared, rained = forecast(rng)
    rows = calibration_table(declared, rained)

    print(f"--- 予報士A（宣言どおりに降る）の {N_DAYS:,} 日（seed={SEED}）---")
    print(f"  {'宣言':>6}{'日数':>7}{'降った':>8}{'実現率':>9}{'ずれ':>9}")
    for level, n, k, rate in rows:
        print(f"  {level:>6.1f}{n:>7}{k:>8}{rate:>9.3f}{rate - level:>9.3f}")

    row70 = next(r for r in rows if r[0] == 0.7)
    print(f"\n  宣言0.7 の日: {row70[1]} 日中 {row70[2]} 日で降った → 実現率 {row70[3]:.3f}")
    se70 = np.sqrt(row70[3] * (1 - row70[3]) / row70[1])
    print(f"  この {row70[1]} 日から測った実現率の標準誤差は ±{1.96 * se70:.3f}。"
          "0.7 はこの幅に入っている")

    brier = float(np.mean((declared - rained) ** 2))
    print(f"\n  Brier スコア {brier:.4f}"
          "（宣言確率と結果(0/1)の二乗誤差の平均。小さいほどよい）")
    print(f"  参考: 毎日「50%」と言うだけの予報士なら {np.mean((0.5 - rained) ** 2):.4f}、"
          f"毎日「{rained.mean():.1f}」なら {np.mean((rained.mean() - rained) ** 2):.4f}")

    print("\n--- 1日だけでは検証できない、という注記 ---")
    day = 0
    print(f"  {day + 1}日目: 宣言 {declared[day]:.1f} → 実際は "
          f"{'降った' if rained[day] else '降らなかった'}")
    print("  この1行から予報の良し悪しは決まらない。宣言 0.1 の日に降ることも、"
          "宣言 0.9 の日に降らないこともある")
    low, high = declared == 0.1, declared == 0.9
    print(f"    宣言0.1 の {int(low.sum())} 日のうち {int(rained[low].sum())} 日は降った")
    print(f"    宣言0.9 の {int(high.sum())} 日のうち {int((~rained[high]).sum())} 日は降らなかった")
    print("  検証できるのは「同じことを何度も言わせたあと」だけ。"
          "確率は1回の結果ではなく、宣言の集まりに対して評価される")

    print("\n--- 比較: 言い過ぎる予報士B（宣言ほどには降らない）---")
    rng_b = np.random.default_rng(SEED + 1)
    dec_b, rain_b = forecast(rng_b, bias=2.0)
    rows_b = calibration_table(dec_b, rain_b)
    row70_b = next(r for r in rows_b if r[0] == 0.7)
    print(f"  宣言0.7 の日: {row70_b[1]} 日中 {row70_b[2]} 日 → 実現率 {row70_b[3]:.3f}"
          f"（0.7 と言いながら実際は {0.7**2:.2f}）")
    print(f"  Brier スコア {np.mean((dec_b - rain_b) ** 2):.4f} ← A より悪い")

    fig, (ax1, ax2) = plots.figure(1, 2, w=2.0, h=0.95)

    # 左: キャリブレーション曲線。対角線が「宣言どおり」。
    lv = np.array([r[0] for r in rows])
    rt = np.array([r[3] for r in rows])
    ns = np.array([r[1] for r in rows])
    ax1.plot([0, 1], [0, 1], color=plots.PALETTE["truth"], lw=1.1, zorder=4)
    ax1.annotate("宣言どおり", xy=(0.36, 0.24), fontsize=6.0,
                 color=plots.PALETTE["truth"], rotation=38)
    se = np.sqrt(np.clip(rt * (1 - rt), 1e-9, None) / ns)
    ax1.errorbar(lv, rt, yerr=1.96 * se, fmt="o", ms=3.2, lw=0.9, capsize=1.6,
                 color=plots.PALETTE["estimate"], zorder=5)
    lv_b = np.array([r[0] for r in rows_b])
    rt_b = np.array([r[3] for r in rows_b])
    ax1.plot(lv_b, rt_b, "s--", ms=2.6, lw=0.9, dashes=(3, 2.0),
             color=plots.PALETTE["alt"], zorder=3)
    ax1.annotate("予報士A", xy=(0.05, 0.92), xycoords="axes fraction", va="top",
                 fontsize=6.2, color=plots.PALETTE["estimate"])
    ax1.annotate("予報士B（言い過ぎ）", xy=(0.05, 0.82), xycoords="axes fraction", va="top",
                 fontsize=6.2, color=plots.PALETTE["alt"])
    ax1.set_xlabel("宣言した確率")
    ax1.set_ylabel("実際に降った割合")
    ax1.set_title("キャリブレーション曲線")

    # 右: 各宣言確率が何日あったか。左の点の信頼性はこの日数で決まる。
    ax2.bar(lv, ns, width=0.07, color=plots.PALETTE["data"], alpha=0.65, lw=0)
    hit = lv == 0.7
    ax2.bar(lv[hit], ns[hit], width=0.07, color=plots.PALETTE["reject"], alpha=0.9, lw=0)
    ax2.annotate(f"宣言0.7 は {row70[1]} 日\n（{row70[2]} 日で降った）",
                 xy=(0.7, ns[hit][0]), xytext=(0, 4), textcoords="offset points",
                 ha="center", fontsize=6.0, color=plots.PALETTE["reject"])
    ax2.set_ylim(0, ns.max() * 1.35)
    ax2.set_xlabel("宣言した確率")
    ax2.set_ylabel("日数")
    ax2.set_title(f"1日では数えられない（Brier {brier:.4f}）")

    fig.tight_layout()
    plots.save(fig, "fig-3-8-calibration.png")


if __name__ == "__main__":
    main()
