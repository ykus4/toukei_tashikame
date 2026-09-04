"""相関は因果ではない、の中身 — 共通原因が作る相関と、層別で反転する相関。

x から y への矢印を一本も引かずに、x と y の相関だけを立てることができる。両方に
影響する z を1つ置けばよい。z を測っていれば、z の値が近いもの同士だけを比べる
（層別する）ことで相関は消える。消えることが、もともと無かったことの確認になる。

さらに厄介なのが Simpson のパラドックスで、層別すると相関の**符号が変わる**。全体で
右上がりに見えるのに、どの層の中でも右下がり。どちらも同じデータの正しい要約であり、
どちらを答えとするかはデータではなく「何を問いたいか」が決める。

相関係数は、どんなに大きくても、この2つの状況と区別がつかない。

    uv run python examples/ch11/confounder_creates_correlation.py
"""

import numpy as np
import pandas as pd
from scipy import stats

from toukei_tashikame import datasets, plots

N, SEED, N_STRATA = 2_000, 116, 4


def strata_table(df: pd.DataFrame, n_strata: int = N_STRATA) -> pd.DataFrame:
    """z の四分位で層に切り、層ごとの相関を並べる。"""
    df = df.assign(層=pd.qcut(df["z"], n_strata, labels=[f"z 第{i + 1}層" for i in range(n_strata)]))
    rows = []
    for name, g in df.groupby("層", observed=True):
        r = float(np.corrcoef(g["x"], g["y"])[0, 1])
        p = float(stats.pearsonr(g["x"], g["y"]).pvalue)
        rows.append({"層": name, "n": len(g), "r": r, "p": p,
                     "傾き": float(np.polyfit(g["x"], g["y"], 1)[0])})
    return pd.DataFrame(rows).set_index("層")


def main() -> None:
    plots.setup()

    print(f"--- 11-6 共通原因だけで相関を立てる（n={N:,}, seed={SEED}）---")
    print("  生成規則: z ~ N(0,1),  x = z + 誤差,  y = 0*x + z + 誤差")
    print("  x から y への矢印は無い（係数はちょうど 0）")
    conf = datasets.confounded_xy(N, effect=0.0, z_strength=1.0, seed=SEED)
    r_all = float(np.corrcoef(conf["x"], conf["y"])[0, 1])
    p_all = float(stats.pearsonr(conf["x"], conf["y"]).pvalue)
    print(f"\n  z を無視した相関   r = {r_all:.4f}   p = {p_all:.2e}   ← 強く「有意」")
    print(f"  x と z の相関 {np.corrcoef(conf['x'], conf['z'])[0, 1]:.4f} / "
          f"y と z の相関 {np.corrcoef(conf['y'], conf['z'])[0, 1]:.4f}")

    tab = strata_table(conf)
    print(f"\n  z を{N_STRATA}層に切って、層の中だけで見ると:")
    print(tab.to_string(float_format=lambda v: f"{v: .4f}"))
    print(f"  全体の {r_all:+.4f} から見れば激減した。ただし端の層はまだ +0.2 近く残っている。")
    print("  層の中でも z は動いているからで（第1層の z は下位25%ぶんの幅を持つ）、これを残差交絡と呼ぶ。")
    print("\n  層を細かくすると消えていく:")
    for k in (4, 10, 20):
        t = strata_table(conf, k)
        print(f"    {k:>2}層（1層 {N // k} 件）  層内 r の平均 {t['r'].mean():+.4f}"
              f"   範囲 [{t['r'].min():+.4f}, {t['r'].max():+.4f}]")
    rx = conf["x"] - np.polyval(np.polyfit(conf["z"], conf["x"], 1), conf["z"])
    ry = conf["y"] - np.polyval(np.polyfit(conf["z"], conf["y"], 1), conf["z"])
    print(f"    z で回帰して残差同士（偏相関）  r = {np.corrcoef(rx, ry)[0, 1]:+.4f}"
          f"   ← 真の効果 0 と整合する")

    print(f"\n--- 11-8 Simpson のパラドックス（n={N:,}, seed={SEED}）---")
    print("  生成規則: y = -1*x + 3z + 誤差。x の係数は負なのに、全体では正に見える")
    simp = datasets.confounded_xy(N, simpson=True, seed=SEED)
    r_simp = float(np.corrcoef(simp["x"], simp["y"])[0, 1])
    stab = strata_table(simp)
    print(f"\n  全体の相関   r = {r_simp:+.4f}（傾き {np.polyfit(simp['x'], simp['y'], 1)[0]:+.4f}）")
    print(stab.to_string(float_format=lambda v: f"{v: .4f}"))
    print(f"  層内の r は {stab['r'].max():+.4f} 以下、傾きは "
          f"{stab['傾き'].mean():+.4f} 前後。全体と符号が逆")
    print("  ← どちらの数字も計算は正しい。全体の +r は「z の違う集団を混ぜた結果」であって、")
    print("    x を1つ増やしたときに y がどうなるかの答えではない")

    print("\n  層別すれば必ず正しくなる、でもない。層別すべき z を知っていることが前提で、")
    print("  それは相関係数からは決して分からない。第17章で扱うのはこの「何で層別するか」の話")

    # --- 図 ---
    fig, axes = plots.figure(1, 3, w=1.95, h=0.95)
    for ax, (df, title) in zip(
        axes[:2],
        [(conf, f"交絡: 全体 r = {r_all:+.4f}"),
         (simp, f"Simpson: 全体 r = {r_simp:+.4f}")],
        strict=True,
    ):
        ax.scatter(df["x"], df["y"], s=3, c=df["z"], cmap="Greys", lw=0, alpha=0.75, zorder=3)
        xs = np.array([df["x"].min(), df["x"].max()])
        b, a = np.polyfit(df["x"], df["y"], 1)
        ax.plot(xs, a + b * xs, color=plots.PALETTE["truth"], lw=1.4, zorder=6)
        # 層ごとの直線を青で重ねる。全体の赤い線と向きを見比べる。
        q = pd.qcut(df["z"], N_STRATA, labels=False)
        for k in range(N_STRATA):
            g = df[q == k]
            bx = np.array([g["x"].quantile(0.02), g["x"].quantile(0.98)])
            bb, aa = np.polyfit(g["x"], g["y"], 1)
            ax.plot(bx, aa + bb * bx, color=plots.PALETTE["estimate"], lw=1.0, zorder=5)
        ax.set_title(f"{title}\n赤 = 全体の回帰、青 = z の層ごと（濃い点ほど z が大）")
        ax.set_xlabel("x")
    axes[0].set_ylabel("y")

    ax = axes[2]
    idx = np.arange(N_STRATA)
    ax.bar(idx - 0.2, tab["r"], width=0.38, color=plots.PALETTE["estimate"], zorder=3)
    ax.bar(idx + 0.2, stab["r"], width=0.38, color=plots.PALETTE["reject"], zorder=3)
    ax.axhline(0, color=plots.PALETTE["ink2"], lw=0.8, zorder=4)
    ax.axhline(r_all, color=plots.PALETTE["truth"], lw=1.0, ls="--", dashes=(4, 2.0), zorder=5)
    ax.axhline(r_simp, color=plots.PALETTE["truth"], lw=1.0, ls=":", zorder=5)
    ax.annotate(f"交絡の全体 r {r_all:+.2f}", xy=(0.02, r_all), xytext=(0, 2),
                textcoords="offset points", fontsize=6.0, color=plots.PALETTE["truth"])
    ax.annotate(f"Simpson の全体 r {r_simp:+.2f}", xy=(0.02, r_simp), xytext=(0, 2),
                textcoords="offset points", fontsize=6.0, color=plots.PALETTE["truth"])
    ax.set_xticks(idx)
    ax.set_xticklabels([f"第{i + 1}層" for i in idx])
    ax.set_ylabel("層内の相関")
    ax.set_title("層内 r（青=交絡, 橙=Simpson）")
    fig.tight_layout()
    plots.save(fig, "fig-11-6-confounding-and-simpson.png")


if __name__ == "__main__":
    main()
