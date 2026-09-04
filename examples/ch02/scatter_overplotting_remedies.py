"""20万点の散布図は塗りつぶされる。ピクセルを数えて、4つの手当てを比べる。

点が多いと散布図はインクで埋まり、密度の情報が落ちる。恐いのは真っ黒になることでは
なく、**4割方埋まった図がまだ図に見えてしまう**ことである。実際にレンダリング結果の
画素を数えて被覆率を出し、透明度・hexbin・間引きがそれをどこまで戻すかを測る。
間引きは相関の推定値もぶらすので、その代償も一緒に見る。

    uv run python examples/ch02/scatter_overplotting_remedies.py
"""

import numpy as np

from toukei_tashikame import datasets, plots

N, RHO, SEED = 200_000, 0.6, 5
N_SAMPLE = 5_000
WINDOW = (-4.5, 4.5)
CORE = (-1.5, 1.5)


def ink(x, y, *, alpha: float = 1.0, size: float = 6.0,
        window: tuple[float, float] = WINDOW) -> tuple[float, float]:
    """描いてから画素を数える。(白でない画素の割合, 平均インク量)。

    「点が何個あるか」ではなく「紙がどれだけ埋まったか」を測りたいので、実際に
    レンダリングした結果を見る。マーカーの大きさも透明度も、ここに効いてくる。
    軸を figure いっぱいに広げてから測るので、余白が混ざらない。
    """
    import matplotlib.pyplot as plt

    fig, ax = plots.figure()
    ax.set_position([0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    ax.set_xlim(*window)
    ax.set_ylim(*window)
    ax.scatter(x, y, s=size, alpha=alpha, color=plots.PALETTE["data"], lw=0,
               rasterized=True)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].astype(float)
    plt.close(fig)
    covered = float((buf < 250).any(axis=-1).mean())
    darkness = float(1.0 - buf.mean() / 255.0)
    return covered, darkness


def main() -> None:
    plots.setup()
    print("--- 2-7 過剰プロットの手当て ---")

    x, y = datasets.bivariate_normal(N, rho=RHO, seed=SEED)
    r_all = float(np.corrcoef(x, y)[0, 1])
    print(f"二変量正規  n={N:,}  真の相関 ρ={RHO}  seed={SEED}")
    print(f"全データの r = {r_all:.4f}（真値からのずれ {r_all - RHO:+.4f}）")

    rng = np.random.default_rng(SEED)
    idx = rng.choice(N, size=N_SAMPLE, replace=False)
    xs, ys = x[idx], y[idx]
    r_sub = float(np.corrcoef(xs, ys)[0, 1])
    print(f"{N_SAMPLE:,}点サンプルの r = {r_sub:.4f}（全データとの差 {r_sub - r_all:+.4f}）")

    # 間引きの誤差は、Fisher の z 変換から出る理論値と突き合わせられる。
    half = 1.96 / np.sqrt(N_SAMPLE - 3)
    z = np.arctanh(r_sub)
    print(f"  n={N_SAMPLE:,} での r の 95% 区間（Fisher z）"
          f" [{np.tanh(z - half):.4f}, {np.tanh(z + half):.4f}]"
          f"  ← 全データの {r_all:.4f} を含む。間引いても相関は読める")

    print(f"\n描画結果の画素を数える（{WINDOW[0]}〜{WINDOW[1]} の窓、433×270 画素）:")
    print(f"{'描き方':<26}{'被覆率':>9}{'平均インク量':>13}")
    for label, xv, yv, kw in [
        (f"生の散布図 {N:,}点", x, y, {}),
        (f"alpha=0.01 {N:,}点", x, y, {"alpha": 0.01}),
        (f"間引き {N_SAMPLE:,}点", xs, ys, {}),
    ]:
        cov, dark = ink(xv, yv, **kw)
        print(f"{label:<26}{cov:>9.4f}{dark:>13.4f}")

    core_cov, _ = ink(x, y, window=CORE)
    print(f"\n  中心部（{CORE[0]}〜{CORE[1]} だけを拡大して描いた場合）の被覆率 "
          f"{core_cov:.4f}")
    print("  ← 密度が高い場所ほど完全に埋まる。つまり散布図が最も情報を落とすのは、"
          "\n    いちばん見たい「点が集まっているところ」である")

    print("\nhexbin は数えてから塗る（六角セルの度数）:")
    import matplotlib.pyplot as plt

    fig_tmp, ax_tmp = plots.figure()
    hb = ax_tmp.hexbin(x, y, gridsize=50)
    counts = np.asarray(hb.get_array(), dtype=float)
    plt.close(fig_tmp)
    nonzero = counts[counts > 0]
    print(f"  gridsize=50 → 全 {counts.size:,} セル中、空でないのは {nonzero.size:,} 個")
    print(f"  最大セル {int(counts.max()):,} 件 / 空でないセルの中央値 "
          f"{np.median(nonzero):.0f} 件 / 平均 {nonzero.mean():.1f} 件")
    print(f"  最も混んだセル1つで全体の {counts.max() / N:.2%}。"
          f"最大と中央値の比は {counts.max() / np.median(nonzero):.0f} 倍")
    print("  ← 散布図が捨てていたのは、この高さの情報である")

    fig, axes = plots.figure(2, 2, h=1.8, w=1.7, sharex=True, sharey=True)
    axes[0, 0].scatter(x, y, s=6, color=plots.PALETTE["data"], lw=0, rasterized=True)
    axes[0, 0].set_title(f"生の散布図（{N:,}点）")
    axes[0, 1].scatter(x, y, s=6, alpha=0.01, color=plots.PALETTE["data"], lw=0,
                       rasterized=True)
    axes[0, 1].set_title("alpha = 0.01")
    axes[1, 0].hexbin(x, y, gridsize=50, cmap="Greys", linewidths=0, rasterized=True,
                      extent=(*WINDOW, *WINDOW))
    axes[1, 0].set_title("hexbin（gridsize=50）")
    axes[1, 1].scatter(xs, ys, s=6, color=plots.PALETTE["data"], lw=0, rasterized=True)
    axes[1, 1].set_title(f"{N_SAMPLE:,}点に間引き  r={r_sub:.4f}")
    for ax in axes.ravel():
        ax.set_xlim(*WINDOW)
        ax.set_ylim(*WINDOW)
    for ax in axes[1]:
        ax.set_xlabel("x")
    for ax in axes[:, 0]:
        ax.set_ylabel("y")
    plots.save(fig, "fig-2-7-overplotting.png")


if __name__ == "__main__":
    main()
