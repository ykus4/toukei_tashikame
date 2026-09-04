"""$R^2$ は y と無関係な変数を足すだけで必ず上がる。1,000回の平均で単調増加を見る。

n=50 のデータに、y とまったく関係のない乱数の列を1本ずつ 30本まで足していく。
$R^2$ は一度も下がらない。下がりようがないからで、変数を1本足したモデルは前のモデルを
（その係数を0にすれば）含んでいる。当てはまりの良さは「含む」関係のぶんだけ必ず改善する。

自由度調整済み $R^2$ は、変数を足したぶんを罰する。こちらは平均としてはほとんど動かず、
母集団の $R^2$ の付近に留まる（そのかわり試行ごとのばらつきは倍近くに膨らむ）。
$R^2$ でモデルを選ぶという行為が何をしているのかは、この2本の曲線が離れていく様子で
一度に分かる。

    uv run python examples/ch12/r2_always_increases_with_noise_vars.py
"""

import numpy as np

from toukei_tashikame import plots

N, MAX_NOISE, TRIALS, SEED = 50, 30, 1_000, 126
B0, B1, SIGMA = 1.0, 1.0, 1.2      # 真は説明変数1本だけ。R² は 0.4 前後になる設定


def one_trial(rng) -> tuple[np.ndarray, np.ndarray]:
    """雑音変数を 0..MAX_NOISE 本足したときの (R², 調整済み R²) を返す。

    入れ子のモデルなので、計画行列を1回 QR 分解すれば全部が出る。直交化した列に
    y を射影した成分の2乗和が、その列までで説明できた平方和である。
    """
    x = rng.normal(0.0, 1.0, size=N)
    y = B0 + B1 * x + rng.normal(0.0, SIGMA, size=N)
    noise = rng.normal(0.0, 1.0, size=(N, MAX_NOISE))     # y と無関係
    X = np.column_stack([np.ones(N), x, noise])

    q, _ = np.linalg.qr(X)
    c = q.T @ y                       # 各直交方向への射影
    ss_tot = float(((y - y.mean()) ** 2).sum())
    # 先頭列（切片）を除いて累積すると、変数 k 本までの説明平方和になる
    ss_model = np.cumsum(c[1:] ** 2)
    r2 = ss_model / ss_tot
    k = np.arange(1, MAX_NOISE + 2)   # 説明変数の本数（真の1本 + 雑音）
    r2_adj = 1 - (1 - r2) * (N - 1) / (N - k - 1)
    return r2, r2_adj


def main() -> None:
    plots.setup()
    rng = np.random.default_rng(SEED)
    r2s, adjs = zip(*(one_trial(rng) for _ in range(TRIALS)), strict=True)
    r2 = np.array(r2s)          # (TRIALS, MAX_NOISE+1)
    adj = np.array(adjs)
    n_noise = np.arange(MAX_NOISE + 1)

    print(f"--- n={N}、真の説明変数1本に、無関係な乱数変数を {MAX_NOISE} 本まで足す"
          f"（{TRIALS:,} 回の平均）---")
    r2_pop = B1**2 / (B1**2 + SIGMA**2)      # 母集団の R²（真値）
    print(f"  母集団の R²（真値）= {r2_pop:.4f}")
    print("  雑音変数の本数     R²      調整済み R²   調整済みの標準偏差   R²>0.85 の割合")
    for k in (0, 5, 10, 20, 30):
        print(f"        {k:>2}          {r2[:, k].mean():.4f}     {adj[:, k].mean():+.4f}"
              f"          {adj[:, k].std(ddof=1):.4f}            "
              f"{float((r2[:, k] > 0.85).mean()):.4f}")

    # 単調増加であることを、1本ずつの差の符号で確かめる
    diffs = np.diff(r2, axis=1)
    print(f"\n  1本足したとき R² が下がった回数: {int((diffs < 0).sum())} /"
          f" {diffs.size:,}（下がりようがない）")
    print(f"  調整済み R² が下がった割合: {float((np.diff(adj, axis=1) < 0).mean()):.4f}")
    print(f"  雑音を {MAX_NOISE} 本入れたときの R² の最大値 {r2[:, -1].max():.4f}"
          f"（n={N} に対し説明変数 {MAX_NOISE + 1} 本）")
    print(f"  真の変数だけのときの R² {r2[:, 0].mean():.4f}"
          f" → 雑音30本で {r2[:, -1].mean():.4f}")
    print(f"  調整済みは {adj[:, 0].mean():.4f} → {adj[:, -1].mean():.4f}")

    print("\n  R² が上がったことは、モデルが良くなった証拠にならない。")
    print("  乱数を足しても上がるのだから、上がったこと自体には情報がない。")
    print("  調整済み R² の平均が動かないのは、それが母集団の R² のほぼ不偏な推定量だから。")
    print(f"  動かないのは平均だけで、ばらつきは {adj[:, 0].std(ddof=1):.4f} →"
          f" {adj[:, -1].std(ddof=1):.4f} に膨らむ。1回の分析では大きく外れうる。")

    # --- 図 ---
    fig, ax = plots.figure(w=1.15)
    pal = plots.PALETTE
    ax.fill_between(n_noise, np.percentile(r2, 10, axis=0), np.percentile(r2, 90, axis=0),
                    color=pal["estimate"], alpha=0.15, lw=0)
    ax.plot(n_noise, r2.mean(axis=0), color=pal["estimate"], lw=1.4)
    ax.annotate(f"$R^2$（{r2[:, 0].mean():.3f} → {r2[:, -1].mean():.3f}）",
                xy=(MAX_NOISE, r2[:, -1].mean()), xytext=(-4, 4),
                textcoords="offset points", ha="right", fontsize=6.2, color=pal["estimate"])
    ax.plot(n_noise, adj.mean(axis=0), color=pal["data"], lw=1.4, ls="--", dashes=(4, 2.0))
    ax.annotate(f"自由度調整済み（{adj[:, 0].mean():.3f} → {adj[:, -1].mean():.3f}）",
                xy=(MAX_NOISE, adj[:, -1].mean()), xytext=(-4, -4),
                textcoords="offset points", ha="right", va="top",
                fontsize=6.2, color=pal["data"])
    plots.mark_truth(ax, r2_pop, f"母集団の $R^2$ = {r2_pop:.3f}", axis="y")
    ax.set_xlabel(f"足した雑音変数の本数（n={N}）")
    ax.set_ylabel(f"{TRIALS:,} 回の平均")
    ax.set_title("無関係な変数でも $R^2$ は必ず上がる")
    fig.tight_layout()
    plots.save(fig, "fig-12-6-r2-vs-noise-vars.png")


if __name__ == "__main__":
    main()
