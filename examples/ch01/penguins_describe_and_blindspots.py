"""実データを要約する。そして、要約が何を隠すかを見る。

Palmer Penguins（seaborn 同梱、344 行）で本書の定型要約を作る。ここまでの道具が実データで
どう動くかの確認が半分、残りの半分は「要約だけを見ていると気づけないこと」の実演である。

体重の全体平均 4,201.75g は、3 種を混ぜた平均であって、どの種の平均でもない。全体の
ヒストグラムは種を混ぜたせいで二山になっている。くちばしの長さと深さの相関にいたっては、
全体で負・種ごとに全部正という符号の反転（シンプソンのパラドックス）まで起きる。

欠測の扱いも隠さない。4 列のいずれかが欠けている 2 行を落として 342 行で解析する。

    uv run python examples/ch01/penguins_describe_and_blindspots.py
"""

import numpy as np
import pandas as pd

from toukei_tashikame import datasets, describe, plots

COLS = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]


def main() -> None:
    plots.setup()
    pd.set_option("display.width", 200)

    import seaborn as sns
    raw = sns.load_dataset("penguins")
    df = datasets.penguins()   # 4列のいずれかが欠けた行を落とす

    print(f"--- 読み込み（seaborn {sns.__version__}）---")
    print(f"  生データ {raw.shape[0]} 行 → 解析対象 {df.shape[0]} 行"
          f"（{raw.shape[0] - df.shape[0]} 行を欠測で除外）")
    na = raw[[*COLS, "sex"]].isna().sum()
    print("  列ごとの欠測数  " + " / ".join(f"{c}={int(na[c])}" for c in na.index))

    print("\n--- 定型要約（describe.summary）---")
    print(describe.summary(df[COLS]).round(2).to_string())

    print("\n--- 体重を種で層別すると ---")
    overall = describe.mean(df["body_mass_g"])
    print(f"{'':<12}{'n':>6}{'平均(g)':>10}{'中央値(g)':>10}{'SD(g)':>10}{'全体平均との差':>14}")
    print(f"{'全体':<12}{len(df):>6}{overall:>10.2f}"
          f"{describe.median(df['body_mass_g']):>10.2f}"
          f"{describe.sd(df['body_mass_g']):>10.2f}{0.0:>14.2f}")
    for name, g in df.groupby("species", observed=True):
        m = describe.mean(g["body_mass_g"])
        print(f"{name:<12}{len(g):>6}{m:>10.2f}{describe.median(g['body_mass_g']):>10.2f}"
              f"{describe.sd(g['body_mass_g']):>10.2f}{m - overall:>14.2f}")
    print("  全体平均はどの種の平均でもない。Gentoo だけが約 875g 重く、"
          "全体平均はその混合比で決まっている")

    print("\n--- 要約の死角: くちばしの長さと深さの相関 ---")
    r_all = float(np.corrcoef(df["bill_length_mm"], df["bill_depth_mm"])[0, 1])
    print(f"{'全体':<12}r = {r_all:+.3f}")
    for name, g in df.groupby("species", observed=True):
        r = float(np.corrcoef(g["bill_length_mm"], g["bill_depth_mm"])[0, 1])
        print(f"{name:<12}r = {r:+.3f}")
    print("  全体では負、種ごとには全部正。層を無視した要約は符号ごと逆を向くことがある")

    fig, axes = plots.figure(1, 2, w=1.6)
    ax = axes[0]
    bins = np.linspace(2500, 6500, 40)
    ax.hist(df["body_mass_g"], bins=bins, color=plots.PALETTE["data"], alpha=0.45, lw=0)
    colors = [plots.PALETTE["estimate"], plots.PALETTE["reject"], plots.PALETTE["alt"]]
    for i, ((name, g), c) in enumerate(
            zip(df.groupby("species", observed=True), colors, strict=True)):
        ax.hist(g["body_mass_g"], bins=bins, histtype="step", color=c, lw=1.1)
        ax.annotate(name, xy=(describe.mean(g["body_mass_g"]), 0.98 - 0.10 * i),
                    xycoords=("data", "axes fraction"), fontsize=6.0, color=c,
                    ha="center", va="top")
    ax.axvline(overall, color=plots.PALETTE["ink"], lw=1.1)
    ax.annotate(f"全体平均 {overall:.0f}g", xy=(overall, 0.45),
                xycoords=("data", "axes fraction"), fontsize=6.0, ha="right", va="center",
                xytext=(-3, 0), textcoords="offset points")
    ax.set_xlabel("体重（g）")
    ax.set_ylabel("羽数")
    ax.set_title("全体平均はどの種の平均でもない")

    ax = axes[1]
    for (_name, g), c in zip(df.groupby("species", observed=True), colors, strict=True):
        ax.scatter(g["bill_length_mm"], g["bill_depth_mm"], s=5, color=c, lw=0, alpha=0.8)
        b, a = np.polyfit(g["bill_length_mm"], g["bill_depth_mm"], 1)
        xs = np.linspace(g["bill_length_mm"].min(), g["bill_length_mm"].max(), 2)
        ax.plot(xs, a + b * xs, color=c, lw=1.0)
    b, a = np.polyfit(df["bill_length_mm"], df["bill_depth_mm"], 1)
    xs = np.linspace(df["bill_length_mm"].min(), df["bill_length_mm"].max(), 2)
    ax.plot(xs, a + b * xs, color=plots.PALETTE["truth"], lw=1.3)
    ax.annotate(f"全体 r={r_all:+.2f}", xy=(xs[1], a + b * xs[1]), fontsize=6.0,
                color=plots.PALETTE["truth"], ha="right", va="bottom")
    ax.set_xlabel("くちばしの長さ（mm）")
    ax.set_ylabel("くちばしの深さ（mm）")
    ax.set_title("全体は負、種ごとは正")

    plots.save(fig, "fig-1-6-penguins-summary-hides-groups.png")


if __name__ == "__main__":
    main()
