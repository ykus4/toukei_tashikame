"""実データ（tips, n=244）に重回帰を当て、残差を4枚組で診断する。

`tip ~ total_bill + size` は係数もp値も申し分なく出る。表だけ見れば何の問題もない。
残差を4枚に描くと話が変わる——支払額が大きいほどチップのばらつきが広がっており（扇形）、
等分散性が破れている。係数そのものは偏らないが、標準誤差とp値は信用できない。

4枚組にするのは、破れ方ごとに見える図が違うからである。1枚だけ見て「問題なし」と
言えないのがこの節の要点で、Breusch-Pagan 検定の p 値と Cook の距離を数字でも添える。

    uv run python examples/ch12/residual_diagnostics_four_panel.py
"""

import numpy as np
from statsmodels.stats.diagnostic import het_breuschpagan

from toukei_tashikame import datasets, describe, plots, regression


def main() -> None:
    plots.setup()
    df = datasets.tips()
    X = np.column_stack([np.ones(len(df)), df["total_bill"], df["size"]])
    y = df["tip"].to_numpy(dtype=float)

    res = regression.ols(X, y, names=["切片", "total_bill", "size"])
    print(f"--- tips（n={len(df)}）に tip ~ total_bill + size を当てる ---")
    print(regression.ols_summary(res))
    print("\n  ここまでは何の問題もなく見える。total_bill も size も有意である。")

    # ① 等分散性 — Breusch-Pagan は「残差の2乗が説明変数で説明できるか」を見る
    lm, lm_p, _, _ = het_breuschpagan(res.resid, X)
    print("\n--- 数字で見る残差 ---")
    print(f"  Breusch-Pagan  LM統計量 {lm:.4f}   p = {lm_p:.4g}")
    print("  ← p が小さい。誤差の大きさが説明変数に依存している（等分散性の破れ）")

    # 実際に扇形になっているかを、当てはめ値の前半・後半で残差の散らばりを比べて確かめる
    order = np.argsort(res.fitted)
    lo_half, hi_half = order[: len(order) // 2], order[len(order) // 2 :]
    print(f"  当てはめ値が小さい半分の残差の標準偏差 {res.resid[lo_half].std(ddof=1):.4f}")
    print(f"  当てはめ値が大きい半分の残差の標準偏差 {res.resid[hi_half].std(ddof=1):.4f}"
          f"   ← {res.resid[hi_half].std(ddof=1) / res.resid[lo_half].std(ddof=1):.2f} 倍に広がる")

    # ② 正規性 — Q-Q の右上が跳ねているかを歪度と尖度で
    print(f"\n  残差の歪度 {describe.skewness(res.resid):.4f}"
          f"   尖度（超過）{describe.kurtosis(res.resid):.4f}   ← 右に長い裾")

    # ③ 影響力 — Cook の距離
    d = regression.cooks_distance(res)
    h = regression.leverage(res)
    top = np.argsort(d)[::-1][:3]
    print(f"\n  Cook の距離の最大 {d.max():.4f}（行番号 {int(np.argmax(d))}）"
          f"   目安 4/n = {4 / len(df):.4f}")
    print("  上位3件（Cook / てこ比 / 残差 / total_bill / tip）")
    for i in top:
        print(f"    行 {i:>3}   {d[i]:.4f}   {h[i]:.4f}   {res.resid[i]:+.3f}"
              f"   {df['total_bill'].iloc[i]:6.2f}   {df['tip'].iloc[i]:5.2f}")
    print(f"  目安を超える点は {(d > 4 / len(df)).sum()} 件。ただし最大でも {d.max():.4f} で、")
    print("  1点で結論がひっくり返るほどではない（0.5 や 1 が目安として使われる）。")

    # ④ 対数変換すると等分散性がどう変わるか
    y_log = np.log(y)
    res_log = regression.ols(np.column_stack([np.ones(len(df)), np.log(df["total_bill"]),
                                              df["size"]]), y_log)
    _, lm_p_log, _, _ = het_breuschpagan(res_log.resid, res_log.X)
    print(f"\n  参考: 両辺を対数にすると Breusch-Pagan の p = {lm_p_log:.4g}"
          f"（R² は {res.r2:.4f} → {res_log.r2:.4f}）")
    print("  変換で等分散性が直ることはあるが、そのぶん係数の意味は「弾力性」に変わる。")

    fig = regression.residual_diagnostics(res, title="tips: tip ~ total_bill + size")
    plots.save(fig, "fig-12-5-residual-four-panel.png")


if __name__ == "__main__":
    main()
