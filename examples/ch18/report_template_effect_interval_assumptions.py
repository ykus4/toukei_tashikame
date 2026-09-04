"""報告文は4項目そろって初めて報告になる — 効果量・区間・仮定・限界。

本書の最後の道具は統計手法ではなく、**書き方の検査器**である。効果量・区間・仮定・
限界の4つのうち1つでも欠けていたら通さない。人のレビューに任せると、忙しい週には
必ず抜ける。抜けるのはいつも「限界」で、そこが読み手にとって最も重要な項目である。

さらに、p 値しか書かれていない報告を警告する。p 値は「効果がどれくらいか」にも
「どれくらい確からしいか」にも答えていないので、これだけを渡された相手は意思決定
できない。この章の依頼1〜5から返ってきた5件の下書きを、そのまま検査にかける。

    uv run python examples/ch18/report_template_effect_interval_assumptions.py
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

REQUIRED = ("effect", "interval", "assumptions", "limitations")
LABELS = {"effect": "効果量", "interval": "区間", "assumptions": "仮定",
          "limitations": "限界"}
# 「効果の大きさ」を名乗るのに必要な最小限の材料。単位か倍率のどちらかは要る。
EFFECT_UNITS = re.compile(r"(pt|%|倍|SD|AUC|d\s*=|h\s*=|OR\s*=|[+-]\s*\d)")
INTERVAL_MARK = re.compile(r"(信頼区間|信用区間|HDI|CI|\[.+,.+\])")
PVALUE_MARK = re.compile(r"(p\s*[=<>]|p\s*値|有意)")


@dataclass
class Report:
    """1件の分析結果。**この4項目がそろわないと報告として出せない。**

    ``assumptions`` と ``limitations`` をリストにしてあるのは、空リストを「書き忘れ」
    ではなく「無いと主張した」と区別せずに落とすため。仮定が0個の分析は存在しない。
    """

    title: str
    section: str
    effect: str = ""
    interval: str = ""
    assumptions: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)

    def check(self) -> dict[str, str]:
        """必須4項目の充足。値が空なら MISSING。"""
        return {k: ("OK" if getattr(self, k) else "MISSING") for k in REQUIRED}

    def warnings(self) -> list[str]:
        """p 値だけで語っている箇所を拾う。数値が p しか無ければ報告にならない。"""
        out = []
        body = f"{self.effect} {self.interval}"
        if PVALUE_MARK.search(body) and not EFFECT_UNITS.search(self.effect):
            out.append("効果量が p 値・有意/非有意の言い換えで書かれている")
        if PVALUE_MARK.search(body) and not INTERVAL_MARK.search(self.interval):
            out.append("区間が無く、不確かさが p 値に置き換わっている")
        return out

    def to_markdown(self) -> str:
        lines = [f"### {self.title}（{self.section}）", ""]
        for key in REQUIRED:
            value = getattr(self, key)
            if not value:
                lines.append(f"- **{LABELS[key]}**: _（未記入）_")
            elif isinstance(value, list):
                lines.append(f"- **{LABELS[key]}**:")
                lines += [f"    - {v}" for v in value]
            else:
                lines.append(f"- **{LABELS[key]}**: {value}")
        return "\n".join(lines)


# --- 依頼1〜5から返ってきた下書き -------------------------------------------
# 数値は同じ章の各スクリプトの出力から引いている。4番目と5番目は、実務でそのまま
# 回ってくる形の下書きにしてある（＝検査器が落とすべきもの）。
def drafts() -> list[Report]:
    return [
        Report(
            title="依頼1 施策Xの効果検証（ランダム化比較）",
            section="18-2",
            effect="CVR 2.97% → 3.28%、差 +0.30pt（相対 +10.1%）、Cohen's h = 0.017",
            interval="差の 95%信頼区間 [-0.46pt, +1.06pt]（p = 0.4407）",
            assumptions=["割付は乱数によるランダム化", "各ユーザの観測は独立",
                         "比率の差に正規近似が効く件数がある"],
            limitations=["設計上 13,914件/群が必要だったが 4,000件/群で打ち切った",
                         "想定した +0.6pt に対する検出力は 0.32 しかない",
                         "「効果がなかった」ではなく「分からなかった」"],
        ),
        Report(
            title="依頼2 施策Yの効果推定（観察データ）",
            section="18-3",
            effect="素朴な群間差 +2.425。共変量調整後は +0.001、IPW では +0.067",
            interval="IPW の 95%信頼区間 [-0.275, +0.409]。E値 6.10（CI下限では 5.73）",
            assumptions=["未観測の交絡が無い（検証不能）", "正値性",
                         "傾向スコアのモデルが正しく指定されている"],
            limitations=["処置はランダムに割り付けられていない",
                         "0/1 から 0.05 以内の傾向スコアが 640件あり、正値性が怪しい",
                         "判定は ANSWER: NOT IDENTIFIED。因果効果として報告できない"],
        ),
        Report(
            title="依頼3 UI-B の採否（ベイズA/B）",
            section="18-4",
            effect="差の事後平均 +0.436pt（相対 +15.1%）、Pr[B > A] = 0.9157",
            interval="94% HDI [-0.175pt, +1.026pt]。B を採って外したときの期待損失 0.0125pt",
            assumptions=["事前分布は Beta(1,1)", "割付はランダム", "各ユーザは独立"],
            limitations=["勝率は事前分布の取り方で動く",
                         "採択閾値（勝率 0.95 / 期待損失 0.05pt）は事業側の合意事項",
                         "見ているのは初回コンバージョンのみ"],
        ),
        # 下書き4: 予測モデルの報告。数字はあるが区間が無く、係数を根拠にしている。
        Report(
            title="依頼4 解約予測モデル",
            section="18-5",
            effect="勾配ブースティングが有意に優れていた（p 値ベースで判断）",
            interval="",
            assumptions=["学習と検証を 7:3 に分割"],
            limitations=["共線性により係数の符号が2個反転している（VIF 最大 14.05）"],
        ),
        # 下書き5: 探索の結果を、そのまま結論として書いてしまった典型。
        Report(
            title="依頼5 セグメント別の傾向",
            section="18-6",
            effect="既存ユーザで指標10 に有意な差が見られた",
            interval="p = 0.0003 で有意",
            assumptions=["データを探索と検証に 50:50 で分割"],
            limitations=[],
        ),
    ]


def fixed_draft4() -> Report:
    """下書き4を、検査を通る形に書き直したもの。直し方まで示すのが 18-7 の趣旨。"""
    return Report(
        title="依頼4 解約予測モデル（修正版）",
        section="18-5",
        effect="検証データの AUC は勾配ブースティング 0.8833 / "
               "ロジスティック回帰 0.8723（差 +0.0110）",
        interval="AUC の差 +0.0110。検証 1,800件では、この差は運用上の優劣を決めない",
        assumptions=["学習4,200件と検証1,800件は無作為に分割",
                     "運用時の分布は学習時と同じ（共変量シフトなし）"],
        limitations=["係数の符号が2個反転しており（VIF 最大 14.05）、"
                     "この係数を施策の根拠に使ってはいけない",
                     "予測性能は介入の効果を意味しない。"
                     "施策効果を言うには第17章の道具が要る"],
    )


def audit(reports: list[Report]) -> tuple[int, int]:
    """全件を検査して、(欠けている項目の数, 警告の出た報告数) を返す。

    状態を先に出して題名を後ろに置く。日本語の題名で桁が揃わないのを避けるためで、
    検査結果の表は「目で走査できること」自体が要件である。
    """
    n_missing, n_flagged = 0, 0
    for r in reports:
        result = r.check()
        n_missing += sum(v == "MISSING" for v in result.values())
        cells = " / ".join(f"{LABELS[k]}: {result[k]}" for k in REQUIRED)
        print(f"  {cells}")
        print(f"      {r.title}")
        warns = r.warnings()
        n_flagged += bool(warns)
        for w in warns:
            print(f"      警告: {w}")
    return n_missing, n_flagged


def main() -> None:
    reports = drafts()

    print("--- 18-1 / 18-7 報告テンプレートの検査 ---\n")
    print(f"  必須項目: {' / '.join(LABELS[k] for k in REQUIRED)}")
    print(f"  対象: 依頼1〜5 の下書き {len(reports)} 件\n")

    print("=== 生成された Markdown 報告文 ===\n")
    for r in reports:
        print(r.to_markdown())
        print()

    print("=== 必須項目チェック ===\n")
    n_missing, n_flagged = audit(reports)

    print(f"\n  欠落 {n_missing} 箇所 / p値だけで語っている報告 {n_flagged} 件")
    for r in reports:
        missing = [LABELS[k] for k, v in r.check().items() if v == "MISSING"]
        if missing:
            print(f"  NG: {r.title} — {'・'.join(missing)} が未記入")
    exit_code = 1 if (n_missing or n_flagged) else 0
    print(f"\n  CI ではここで exit {exit_code}。"
          "報告文は成果物なので、テストと同じ扱いで落とす")

    print("\n=== 直し方（下書き4を書き直す）===\n")
    fixed = fixed_draft4()
    print(fixed.to_markdown())
    print()
    result = fixed.check()
    print(f"  再検査: {' / '.join(f'{LABELS[k]}: {result[k]}' for k in REQUIRED)}"
          f"、警告 {len(fixed.warnings())} 件")
    print("  変えたのは中身ではなく書き方だけ。同じ分析結果でも、"
          "受け取った側にできることが変わる")

    print("\n--- まとめ ---")
    print("  p 値は「効果がどれくらいか」に答えない。効果量が要る")
    print("  効果量は「どれくらい確からしいか」に答えない。区間が要る")
    print("  区間は「何を前提にした区間か」に答えない。仮定が要る")
    print("  仮定は「それが崩れたら何が起きるか」に答えない。限界が要る")
    print("  4つ書いて、はじめて相手が自分で判断できる。それが報告である")

    # exit 1 は実演として上に印字するだけにする。examples は CI が毎回通すので、
    # ここで終了コードを立てると、検査器が動いたこと自体がビルド失敗になる。


if __name__ == "__main__":
    main()
