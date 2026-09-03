"""図の共通スタイル。色は役割で決まり、呼び出し側では選べない。

OUTLINE §6-3 の配色規約をコードで固定する。同じ役割のものが章によって違う色で出ると、
読者は毎回凡例を読み直すことになる。逆に固定してあれば、赤い実線を見た瞬間に「真値だ」
と分かる。だから ``examples/`` の中で色を書かない。役割名で引く。

赤と緑に対立する意味を持たせない。色覚特性によらず読めるようにするためで、対になる
ものは色以外（実線と破線、塗りと枠）でも区別できるようにしてある。

そして**凡例に頼らず、図の中に直接ラベルを置く。** A5 の紙面では凡例の枠が図の面積を
食い、線と凡例のあいだで視線が往復する。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

__all__ = [
    "FIGDIR",
    "PALETTE",
    "coverage_stripes",
    "figure",
    "mark_interval",
    "mark_truth",
    "null_vs_alt",
    "prior_posterior",
    "qq",
    "save",
    "setup",
    "sim_hist",
]

# 出力先。リポジトリルート基準で解決するので、どこから実行しても同じ場所に出る。
FIGDIR = Path(__file__).resolve().parents[2] / "figures"

# A5 の本文段幅は 110mm = 4.33in。ここを基準にしておくと、紙に貼ったときの文字の
# 大きさが図ごとに変わらない。
FIGSIZE = (4.33, 2.7)
DPI = 300

PALETTE = {
    "data": "#585858",       # データ・標本 — グレー
    "truth": "#c8342b",      # 真の値・母数 — 赤の実線。必ず注釈つき
    "estimate": "#1f6fb8",   # 推定値・当てはめ — 青
    "interval": "#1f6fb8",   # 区間 — 青の帯（α=0.2 で塗る）
    "null": "#9a9a9a",       # 帰無分布 — グレーの塗り
    "alt": "#7a4ea3",        # 対立分布 — 紫
    "reject": "#e07b12",     # 棄却域・p値の領域 — オレンジの塗り
    "prior": "#7fb3e0",      # 事前分布 — 薄い青の破線
    "posterior": "#1f6fb8",  # 事後分布 — 濃い青の実線
    "ink": "#0b0b0b",
    "ink2": "#52514e",
    "grid": "#e1e0d9",
}

INTERVAL_ALPHA = 0.20
FILL_ALPHA = 0.35


def setup() -> None:
    """バックエンド・日本語フォント・本書の rcParams を決める。

    ``matplotlib.pyplot`` を最初に触る前に呼ぶこと。フォント探索を後から変えても、
    すでに作られた Figure には効かない。
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    installed = {f.name for f in font_manager.fontManager.ttflist}
    wanted = ["Hiragino Sans", "Noto Sans CJK JP", "Noto Sans JP", "IPAexGothic",
              "Yu Gothic", "DejaVu Sans"]
    family = [f for f in wanted if f in installed]
    if not any(f != "DejaVu Sans" for f in family):
        print("warning: 日本語フォントが見つからない。Noto Sans CJK JP を入れること",
              file=sys.stderr)

    plt.rcParams.update({
        "font.family": family or ["DejaVu Sans"],
        "font.size": 7,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.edgecolor": "#8a8a8a",
        "axes.labelcolor": PALETTE["ink2"],
        "axes.linewidth": 0.6,
        "axes.titlesize": 7,
        "axes.titlelocation": "left",
        "grid.color": PALETTE["grid"],
        "grid.linewidth": 0.5,
        "legend.fontsize": 6.2,
        "legend.frameon": False,
        "lines.linewidth": 1.0,
        "text.color": PALETTE["ink"],
        "xtick.color": "#8a8a8a",
        "ytick.color": "#8a8a8a",
        "xtick.labelsize": 6.2,
        "ytick.labelsize": 6.2,
        "axes.unicode_minus": False,
        "figure.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def figure(rows: int = 1, cols: int = 1, *, h: float = 1.0, w: float = 1.0, **kw):
    """A5 段幅に合わせた Figure を作る。``examples/`` はこれを使う。

    ``plt.subplots`` を直に呼ばれると、図ごとに寸法が変わって紙面で文字の大きさが
    揃わなくなる。増やしたい形があれば、呼び出し側ではなくここに足す。
    """
    import matplotlib.pyplot as plt

    return plt.subplots(rows, cols, figsize=(FIGSIZE[0] * w, FIGSIZE[1] * h), **kw)


def mark_truth(ax, value: float, label: str | None = None, *, axis: str = "x") -> None:
    """真値を赤の実線で引き、図の中に直接ラベルを置く。

    凡例にしないのは、「どれが真値か」は図を見た最初の1秒で分かるべきだから。
    """
    if axis == "x":
        ax.axvline(value, color=PALETTE["truth"], lw=1.1, zorder=5)
        ax.annotate(label or f"真値 = {value:g}", xy=(value, 0.98), xycoords=("data", "axes fraction"),
                    ha="left", va="top", fontsize=6.0, color=PALETTE["truth"], zorder=6,
                    xytext=(3, 0), textcoords="offset points",
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.0})
    else:
        ax.axhline(value, color=PALETTE["truth"], lw=1.1, zorder=5)
        ax.annotate(label or f"真値 = {value:g}", xy=(0.99, value), xycoords=("axes fraction", "data"),
                    ha="right", va="bottom", fontsize=6.0, color=PALETTE["truth"], zorder=6,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.0})


def mark_interval(ax, lo: float, hi: float, *, label: str | None = None, y: float | None = None) -> None:
    """区間を青の帯で示す。``y`` を渡せばその高さの水平線として描く。"""
    if y is None:
        ax.axvspan(lo, hi, color=PALETTE["interval"], alpha=INTERVAL_ALPHA, lw=0, zorder=1)
    else:
        ax.plot([lo, hi], [y, y], color=PALETTE["interval"], lw=1.6, solid_capstyle="butt", zorder=4)
    if label:
        ax.annotate(label, xy=((lo + hi) / 2, 0.02), xycoords=("data", "axes fraction"),
                    ha="center", va="bottom", fontsize=6.0, color=PALETTE["estimate"])


def null_vs_alt(ax, null_x, null_y, alt_x=None, alt_y=None, crit: float | None = None,
                *, tail: str = "upper") -> None:
    """帰無（グレーの塗り）・対立（紫）・棄却域（オレンジの塗り）を1枚に重ねる。

    検出力の図は、この3つが同じ横軸に乗っていないと読めない。α と β が同じ絵の
    別々の面積として見えることが、この図の唯一の仕事である。
    """
    ax.fill_between(null_x, null_y, color=PALETTE["null"], alpha=FILL_ALPHA, lw=0, zorder=1)
    ax.plot(null_x, null_y, color=PALETTE["null"], lw=1.0, zorder=3)
    if alt_x is not None:
        ax.plot(alt_x, alt_y, color=PALETTE["alt"], lw=1.2, zorder=3)
    if crit is not None:
        mask = null_x >= crit if tail == "upper" else null_x <= crit
        ax.fill_between(np.asarray(null_x)[mask], np.asarray(null_y)[mask],
                        color=PALETTE["reject"], alpha=0.55, lw=0, zorder=2)
        ax.axvline(crit, color=PALETTE["reject"], lw=0.9, ls="--", dashes=(4, 2.2), zorder=4)


def coverage_stripes(ax, intervals, truth: float, *, n_show: int = 100) -> int:
    """区間を縦に積んだ縞模様。真値を外した区間だけ赤にする。

    「95%の確率で真値が入る」が誤りであることを、1枚で見せるための図。動いているのは
    区間のほうで、真値は縦の直線として一度も動かない。返り値は外した本数。
    """
    iv = np.asarray(intervals, dtype=float)[:n_show]
    missed = (iv[:, 1] < truth) | (iv[:, 0] > truth)
    for i, ((lo, hi), miss) in enumerate(zip(iv, missed, strict=True)):
        ax.plot([lo, hi], [i, i], lw=0.8, solid_capstyle="butt",
                color=PALETTE["truth"] if miss else PALETTE["interval"],
                alpha=1.0 if miss else 0.55, zorder=3 if miss else 2)
    mark_truth(ax, truth, f"真値 = {truth:g}")
    ax.set_ylim(-1, len(iv))
    ax.set_yticks([])
    ax.set_ylabel(f"{len(iv)} 本の区間")
    return int(missed.sum())


def sim_hist(ax, values, theory=None, *, bins: int = 40, label: str = "シミュレーション",
             theory_label: str = "理論値") -> None:
    """シミュレーションのヒストグラムに、理論の密度を赤の破線で重ねる。

    本書の基本形。数え上げた結果と、紙の上で解いた式が同じ形になることを、毎回この
    かたちで確かめる。``theory`` は ``(x, density)`` のタプル。
    """
    ax.hist(values, bins=bins, density=True, color=PALETTE["data"], alpha=0.55, lw=0)
    if theory is not None:
        tx, ty = theory
        ax.plot(tx, ty, color=PALETTE["truth"], lw=1.2, ls="--", dashes=(4, 2.0), zorder=5)
        ax.annotate(theory_label, xy=(tx[int(len(tx) * 0.72)], ty[int(len(ty) * 0.72)]),
                    fontsize=6.0, color=PALETTE["truth"], ha="left", va="bottom",
                    xytext=(2, 2), textcoords="offset points")


def qq(ax, x, dist: str = "norm") -> None:
    """QQプロット。参照直線は赤の実線。"""
    from scipy import stats

    x = np.sort(np.asarray(x, dtype=float).ravel())
    n = x.size
    # Blom のプロット位置。小標本で端点が潰れにくい。
    p = (np.arange(1, n + 1) - 0.375) / (n + 0.25)
    q = stats.norm.ppf(p) if dist == "norm" else getattr(stats, dist).ppf(p)
    ax.scatter(q, x, s=6, color=PALETTE["data"], lw=0, zorder=3)
    lo, hi = float(q.min()), float(q.max())
    slope, intercept = np.polyfit(q, x, 1)
    ax.plot([lo, hi], [slope * lo + intercept, slope * hi + intercept],
            color=PALETTE["truth"], lw=1.0, zorder=4)
    ax.set_xlabel("理論分位点")
    ax.set_ylabel("標本分位点")


def prior_posterior(ax, grid, prior=None, likelihood=None, posterior=None) -> None:
    """事前（薄い青の破線）・尤度（グレー）・事後（濃い青の実線）を重ねる。"""
    if prior is not None:
        ax.plot(grid, prior, color=PALETTE["prior"], lw=1.1, ls="--", dashes=(4, 2.0),
                zorder=3)
    if likelihood is not None:
        ax.plot(grid, likelihood, color=PALETTE["data"], lw=1.0, zorder=3)
    if posterior is not None:
        ax.plot(grid, posterior, color=PALETTE["posterior"], lw=1.4, zorder=4)
        ax.fill_between(grid, posterior, color=PALETTE["posterior"],
                        alpha=INTERVAL_ALPHA, lw=0, zorder=1)


def save(fig, name: str) -> Path:
    """``figures/<name>`` に 300dpi で書き出す。

    出力先はリポジトリルート基準で解決するので、``examples/ch07/`` から実行しても
    リポジトリルートから実行しても同じ場所に出る。
    """
    import matplotlib.pyplot as plt

    FIGDIR.mkdir(exist_ok=True)
    if not name.endswith(".png"):
        name += ".png"
    out = FIGDIR / name
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"wrote {out}")
    return out
