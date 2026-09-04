"""傾向スコアを推定して、IPW とマッチングを手で書く。極端な重みがどこで牙をむくか。

傾向スコア $e(x) = \\Pr[T=1 \\mid X=x]$ は「どれくらい処置されやすかったか」で、
これで条件づけると、観測された共変量による交絡が落ちる（と仮定する）。使い道は2つ。
重みにするのが IPW、相手を探すのがマッチングである。どちらも 10 行ほどで書ける。

同じ傾向スコアから2通りの推定値が出るが、両者は同じものを推していない。IPW は母集団
全体の ATE、最近傍マッチングは処置群についての ATT を推している。そして IPW には
致命的な弱点がある——$e(x)$ が 0 や 1 に近い人の重みが爆発し、推定値が数人に支配される。

    uv run python examples/ch17/propensity_score_ipw_and_matching.py
"""

import numpy as np

from toukei_tashikame import causal, datasets, plots

N, ATE, CONFOUNDING, SEED = 2000, 1.5, 1.5, 175


def ipw_by_hand(y, t, ps, normalize: bool = True, stabilized: bool = False) -> float:
    """逆確率重み付け。処置されにくかったのに処置された人を重く数える。

    ``normalize=False`` が素の Horvitz–Thompson、``True`` が重みの合計で割る Hajek。
    ``stabilized`` は周辺確率を掛ける安定化重みで、Hajek では分母と約分して消える。
    """
    w = np.where(t == 1, 1.0 / ps, 1.0 / (1.0 - ps))
    if stabilized:
        p = t.mean()
        w = w * np.where(t == 1, p, 1.0 - p)
    if not normalize:
        return float(np.mean(w * t * y) - np.mean(w * (1 - t) * y))
    m1 = np.sum(w * t * y) / np.sum(w * t)
    m0 = np.sum(w * (1 - t) * y) / np.sum(w * (1 - t))
    return float(m1 - m0)


def matching_by_hand(y, t, ps) -> tuple[float, int]:
    """最近傍マッチング（復元あり）。処置群の各人に、傾向スコアが最も近い対照を1人。"""
    treated = np.flatnonzero(t == 1)
    control = np.flatnonzero(t == 0)
    j = np.argmin(np.abs(ps[treated][:, None] - ps[control][None, :]), axis=1)
    diffs = y[treated] - y[control[j]]
    return float(diffs.mean()), diffs.size


def draw(ps, t, smd_before, smd_after, names) -> None:
    fig, axes = plots.figure(1, 2, w=2.0)
    pal = plots.PALETTE

    ax = axes[0]
    bins = np.linspace(0, 1, 45)
    ax.hist(ps[t == 0], bins=bins, color=pal["data"], alpha=0.55, lw=0)
    ax.hist(ps[t == 1], bins=bins, color=pal["estimate"], alpha=0.55, lw=0)
    ax.annotate("対照群", xy=(0.16, 0.86), xycoords="axes fraction", fontsize=6.2,
                color=pal["data"])
    ax.annotate("処置群", xy=(0.74, 0.86), xycoords="axes fraction", fontsize=6.2,
                color=pal["estimate"])
    ax.set_xlabel("推定された傾向スコア $\\hat e(x)$")
    ax.set_ylabel("人数")

    ax = axes[1]
    pos = np.arange(len(names))
    ax.scatter(np.abs(smd_before), pos, s=18, color=pal["reject"], zorder=4, lw=0)
    ax.scatter(np.abs(smd_after), pos, s=18, color=pal["estimate"], zorder=4, lw=0)
    for i in pos:
        ax.plot([abs(smd_before[i]), abs(smd_after[i])], [i, i], color=pal["grid"],
                lw=1.0, zorder=2)
    ax.axvline(0.1, color=pal["truth"], lw=1.0, zorder=3)
    ax.annotate("目安 0.1", xy=(0.1, 0.02), xycoords=("data", "axes fraction"),
                fontsize=6.0, color=pal["truth"], xytext=(3, 0), textcoords="offset points")
    ax.set_yticks(pos)
    ax.set_yticklabels(names, fontsize=6.2)
    ax.set_xlabel("|標準化平均差|  オレンジ＝重み付け前 / 青＝IPW 後")
    fig.tight_layout()
    plots.save(fig, "fig-17-7-propensity-balance.png")


def main() -> None:
    plots.setup()
    d = datasets.observational(n=N, ate=ATE, confounding=CONFOUNDING, seed=SEED)
    y, t, z = d.y, d.x, d.z

    ps = causal.propensity_score(z[:, None], t)     # ロジスティック回帰（IRLS）
    w_raw = np.where(t == 1, 1.0 / ps, 1.0 / (1.0 - ps))

    print(f"--- 傾向スコアで交絡を落とす（n={N:,}, 真の ATE = {ATE}, seed={SEED}）---\n")
    print(f"  推定された傾向スコアの範囲: [{ps.min():.4f}, {ps.max():.4f}]")
    print(f"  0.05 未満 {int((ps < 0.05).sum())}人 / 0.95 超 {int((ps > 0.95).sum())}人"
          "  ← ここに重みの爆弾がある")

    naive = causal.naive_diff(y, t)
    ipw_ht = ipw_by_hand(y, t, ps, normalize=False)
    ipw_hajek = ipw_by_hand(y, t, ps)
    ipw_stab = ipw_by_hand(y, t, ps, stabilized=True)
    match_est, n_pairs = matching_by_hand(y, t, ps)

    print(f"\n{'推定量':<30}{'推定値':>9}{'真値とのずれ':>14}")
    print(f"{'素朴な群間差':<30}{naive.estimate:>9.3f}{naive.estimate - ATE:>+14.3f}")
    print(f"{'IPW（Horvitz–Thompson）':<30}{ipw_ht:>9.3f}{ipw_ht - ATE:>+14.3f}")
    print(f"{'IPW（Hajek＝重みで正規化）':<30}{ipw_hajek:>9.3f}{ipw_hajek - ATE:>+14.3f}")
    print(f"{'安定化 IPW':<30}{ipw_stab:>9.3f}{ipw_stab - ATE:>+14.3f}")
    print(f"{'最近傍マッチング（ATT）':<30}{match_est:>9.3f}{match_est - ATE:>+14.3f}")
    print(f"{'真の ATE':<30}{ATE:>9.3f}{0.0:>+14.3f}")
    print(f"\n  安定化 IPW と Hajek の差は {ipw_stab - ipw_hajek:+.6f}。"
          "周辺確率は分子と分母で約分されるので、")
    print("  正規化してしまえば「安定化」の効果は消える。効くのは正規化しない HT のほう。")
    print(f"  マッチングは {n_pairs:,} 組。狙っているのは ATT なので、"
          "効果に個人差があれば ATE とは別の量になる。")

    lib_ipw = causal.ipw_ate(y, t, ps, stabilized=True)
    lib_match = causal.match_ate(y, t, ps, caliper=0.2)
    print(f"\n  照合: causal.ipw_ate = {lib_ipw.estimate:.3f}"
          f"（手書きとの差 {lib_ipw.estimate - ipw_stab:+.4f}）"
          f" / causal.match_ate = {lib_match.estimate:.3f}")
    print("  道具はトリムやキャリパーを既定で効かせているので、手書きと完全一致はしない。")

    print("\n--- 重みの分布と、極端な重みの影響 ---\n")
    order = np.argsort(w_raw)[::-1]
    print(f"  重みの最大 {w_raw.max():.1f}、上位10人の合計 {w_raw[order[:10]].sum():.1f}"
          f"（全体 {w_raw.sum():.1f} の {w_raw[order[:10]].sum() / w_raw.sum():.1%}）")
    print(f"  最も重い1人: 傾向スコア {ps[order[0]]:.4f}、処置 {int(t[order[0]])}、"
          f"重み {w_raw[order[0]]:.1f} … 実質 {w_raw[order[0]]:.0f}人ぶんとして数えられる")

    print(f"\n{'処理':<34}{'推定値':>9}{'使った人数':>12}{'最大重み':>10}")
    for label, lo, hi in (("トリムなし", 0.0, 1.0), ("傾向スコアを [0.05, 0.95] に制限", 0.05, 0.95),
                          ("[0.10, 0.90] に制限", 0.10, 0.90)):
        keep = (ps > lo) & (ps < hi)
        est = ipw_by_hand(y[keep], t[keep], ps[keep])
        wk = np.where(t[keep] == 1, 1 / ps[keep], 1 / (1 - ps[keep]))
        print(f"{label:<34}{est:>9.3f}{int(keep.sum()):>12,}{wk.max():>10.1f}")
    print("\n  トリムは推定を安定させるが、ただでは済まない。落とした人は"
          "「そもそも処置されえない人」で、")
    print("  推定対象が母集団全体から「重なりのある層」へ静かにすり替わっている。")

    X = np.column_stack([z, z**2])
    names = ["Z", "Z²"]
    before = causal.balance_table(X, t, names=names)
    after = causal.balance_table(X, t, weights=w_raw, names=names)
    print("\n--- 共変量バランス（標準化平均差, |SMD|<0.1 が目安）---\n")
    print(f"{'変数':<6}{'重み付け前':>12}{'IPW 後':>12}")
    for name in names:
        print(f"{name:<6}{before.loc[name, 'SMD']:>12.4f}{after.loc[name, 'SMD']:>12.4f}")
    print("\n  揃った。ただし揃ったのは**表に載っている変数**だけである。")
    print("  未観測の交絡が残っていても、この表は同じように綺麗になる。")
    print("  傾向スコアは条件付き独立の仮定を検証する道具ではなく、仮定の上で計算する道具。")

    draw(ps, t, before["SMD"].to_numpy(), after["SMD"].to_numpy(), names)


if __name__ == "__main__":
    main()
