"""依頼5「何か言えますか？」— 探索で見つけた仮説は、別のデータで確かめるまで仮説。

「何か言えますか」に答えようとすると、指標を全部・セグメントを全部なめることになる。
20指標 × 3セグメント = 60通り試せば、真の効果がゼロでも 5% すなわち3つ前後は
p < 0.05 になる。これは不正ではなく、60回引けばそうなる、というだけの話である。

だから探索と検証にデータを割る。探索側で拾った仮説を、一度も見ていないホールドアウト
に当て直す。生き残らなければ、それは仮説ですらなかった。**同じデータで検定し直しても
何も検証していない**ことを、同じ画面に並べて見せる。

    uv run python examples/ch18/case5_exploratory_then_confirm_on_holdout.py
"""

import numpy as np
from scipy import stats

from toukei_tashikame import plots, sim, testing

N, N_METRIC, SEED = 6000, 20, 186
ALPHA = 0.05
SEGMENTS = {"全体": None, "新規": 1, "既存": 0}
TRIALS = 200          # 分け方を変えて、この手続きの再現率そのものを数える


def make_data(rng: np.random.Generator):
    """真の効果がゼロのデータ。群間差はどの指標にもどのセグメントにも無い。"""
    group = rng.integers(0, 2, size=N)          # 施策あり/なし（無作為割付）
    is_new = rng.integers(0, 2, size=N)         # セグメント分け用の属性
    y = rng.normal(size=(N, N_METRIC))          # 20指標。すべて群と無関係
    return group, is_new, y


def scan(idx, group, is_new, y) -> dict[tuple[str, int], float]:
    """指標 × セグメントを総なめして p 値を返す。これが「探索」の実体。"""
    out: dict[tuple[str, int], float] = {}
    for name, value in SEGMENTS.items():
        sub = idx if value is None else idx[is_new[idx] == value]
        a, b = y[sub][group[sub] == 1], y[sub][group[sub] == 0]
        p = stats.ttest_ind(a, b, axis=0, equal_var=False).pvalue
        for j in range(N_METRIC):
            out[(name, j)] = float(p[j])
    return out


def one_split(rng: np.random.Generator) -> tuple[float, float]:
    """探索→検証を1回。拾った件数と、ホールドアウトで生き残った件数を返す。"""
    group, is_new, y = make_data(rng)
    perm = rng.permutation(N)
    explore, holdout = perm[: N // 2], perm[N // 2:]
    pe = scan(explore, group, is_new, y)
    picked = [k for k, v in pe.items() if v < ALPHA]
    if not picked:
        return 0.0, 0.0
    ph = scan(holdout, group, is_new, y)
    return float(len(picked)), float(sum(ph[k] < ALPHA for k in picked))


def draw(keys, pe, ph, picked) -> None:
    fig, axes = plots.figure(1, 2, w=2.0, h=0.95)
    pal = plots.PALETTE

    # ① 探索側の p 値を小さい順に。60本のうち下端の数本が閾値を割る。
    ax = axes[0]
    order = np.argsort([pe[k] for k in keys])
    vals = np.array([pe[keys[i]] for i in order])
    colors = [pal["reject"] if v < ALPHA else pal["data"] for v in vals]
    ax.scatter(np.arange(len(vals)), vals, s=9, c=colors, lw=0, zorder=4)
    ax.axhline(ALPHA, color=pal["reject"], lw=0.9, ls="--", dashes=(4, 2.2), zorder=5)
    ax.annotate(f"α = {ALPHA}", xy=(0.98, ALPHA), xycoords=("axes fraction", "data"),
                xytext=(0, 3), textcoords="offset points", ha="right", fontsize=6.0,
                color=pal["reject"])
    ax.annotate(f"{len(picked)}件が閾値を割る\n（真の効果はすべてゼロ）",
                xy=(0.06, 0.92), xycoords="axes fraction", fontsize=6.2,
                color=pal["ink2"], va="top")
    ax.set_xlabel(f"{len(keys)} 個の仮説（p の小さい順）")
    ax.set_ylabel("探索データでの p 値")
    ax.set_title("① 総なめすれば、必ず何かが有意になる")

    # ② 探索 vs ホールドアウト。左下の象限だけが「再現した」。
    ax = axes[1]
    x = np.array([pe[k] for k in keys])
    yv = np.array([ph[k] for k in keys])
    ax.scatter(x, yv, s=9, color=pal["data"], lw=0, alpha=0.6, zorder=3)
    sel = np.array([k in picked for k in keys])
    ax.scatter(x[sel], yv[sel], s=22, color=pal["reject"], lw=0, zorder=5)
    ax.axhline(ALPHA, color=pal["reject"], lw=0.8, ls="--", dashes=(4, 2.2), zorder=4)
    ax.axvline(ALPHA, color=pal["reject"], lw=0.8, ls="--", dashes=(4, 2.2), zorder=4)
    ax.annotate("再現した\n（左下）", xy=(0.02, 0.02), xycoords="axes fraction",
                fontsize=6.0, color=pal["estimate"], va="bottom")
    ax.annotate("探索でだけ有意\n＝偽陽性", xy=(0.02, 0.55), xycoords="axes fraction",
                fontsize=6.0, color=pal["reject"], va="bottom")
    ax.set_xlabel("探索データでの p 値")
    ax.set_ylabel("ホールドアウトでの p 値")
    ax.set_title("② 拾った仮説（橙）は左下に来ない")

    fig.tight_layout()
    plots.save(fig, "fig-18-6-explore-then-confirm.png")


def main() -> None:
    plots.setup()
    rng = np.random.default_rng(SEED)
    group, is_new, y = make_data(rng)
    perm = rng.permutation(N)
    explore, holdout = perm[: N // 2], perm[N // 2:]

    n_hyp = N_METRIC * len(SEGMENTS)
    print(f"--- 18-6 依頼5「何か言えますか？」（n={N:,}, 真の効果はすべて0, "
          f"seed={SEED}）---\n")
    print(f"  指標 {N_METRIC} × セグメント {len(SEGMENTS)}"
          f"（{'・'.join(SEGMENTS)}）= {n_hyp} 通りの仮説")
    print(f"  探索 {explore.size:,}件 / ホールドアウト {holdout.size:,}件（50:50）")
    print(f"  α={ALPHA} なら、効果がゼロでも {n_hyp} × {ALPHA} = "
          f"{n_hyp * ALPHA:.0f} 件前後は有意になる計算")

    pe = scan(explore, group, is_new, y)
    keys = list(pe)
    picked = [k for k in keys if pe[k] < ALPHA]

    print("\n① 探索（データを見て仮説を作る）")
    print(f"  有意になった仮説 {len(picked)} / {n_hyp}"
          f"（最小 p = {min(pe.values()):.4f}）")
    for k in picked:
        print(f"    セグメント「{k[0]}」× 指標{k[1]:02d}   p = {pe[k]:.4f}")

    pv = np.array([pe[k] for k in keys])
    holm = testing.adjust_pvalues(pv, "holm")
    bh = testing.adjust_pvalues(pv, "bh")
    print(f"  多重比較補正をかけると Holm {int((holm < ALPHA).sum())}件 / "
          f"BH {int((bh < ALPHA).sum())}件が残る")
    print("  ← 補正は保険であって検証ではない。残ったものが正しいとは言っていない")

    ph = scan(holdout, group, is_new, y)
    replicated = [k for k in picked if ph[k] < ALPHA]
    print("\n② 検証（ホールドアウトは一度も見ていない）")
    print("  仮説               探索の p   ホールドアウトの p   判定")
    for k in picked:
        verdict = "再現した" if ph[k] < ALPHA else "再現しない"
        print(f"  「{k[0]}」× 指標{k[1]:02d}     {pe[k]:>8.4f}         {ph[k]:>8.4f}   {verdict}")
    print(f"  再現 {len(replicated)} / {len(picked)}")

    print("\n③ やってはいけない比較 — 同じデータで検定し直す")
    for k in picked:
        print(f"  「{k[0]}」× 指標{k[1]:02d}   探索データで再検定 p = {pe[k]:.4f}"
              f"（有意）")
    print(f"  {len(picked)} / {len(picked)} が有意。当たり前で、同じ数字を2回見ただけ。")
    print("  「もう一度検定したら有意でした」は検証ではない。"
          "データを分けていないなら、何も確かめていない")

    # --- ④ 手続きそのものの性質を数える ---
    with sim.Timer(f"  {TRIALS}回の分割"):
        out = sim.repeat(one_split, trials=TRIALS, seed=1860, progress=False)
    n_picked, n_repl = out[:, 0], out[:, 1]
    rate = float(n_repl.sum() / max(n_picked.sum(), 1))
    print(f"\n④ データを作り直して {TRIALS} 回まわす（真の効果はいつもゼロ）")
    print(f"  探索で拾う件数の平均       {n_picked.mean():.2f} / {n_hyp}"
          f"（理論値 {n_hyp * ALPHA:.1f}）")
    print(f"  ホールドアウトで残る件数    {n_repl.mean():.2f}")
    print(f"  再現率                    {rate:.4f}"
          f"   ← α={ALPHA} とほぼ同じ。拾ったものはすべて偽陽性だった")
    print(f"  1件でも再現してしまう割合   {(n_repl > 0).mean():.4f}")
    print("  ← ホールドアウトは万能ではない。だが探索の 100% を "
          f"{rate:.0%} まで落とす")

    print("\n⑤ 報告（18-7 のテンプレート）")
    print(f"  効果量: 探索データで最も強かったのは「{picked[0][0]}」× 指標"
          f"{picked[0][1]:02d}（p={pe[picked[0]]:.4f}）")
    print(f"  区間  : ホールドアウトでの p は {ph[picked[0]]:.4f}。"
          "効果は 0 と区別できない")
    print(f"  仮定  : 探索とホールドアウトは無作為に 50:50 に分割／"
          f"{n_hyp} 通りを事前に列挙してある")
    print("  限界  : **これは探索の結果であり、仮説である。**"
          "確認するには新しく取ったデータで、事前登録した1つの仮説だけを検定する")

    draw(keys, pe, ph, set(picked))


if __name__ == "__main__":
    main()
