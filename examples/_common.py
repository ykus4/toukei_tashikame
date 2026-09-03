"""examples 共通の下ごしらえ。図の初期化と出力先の解決だけを置く。

**統計処理をここに書かない。** 本文のコードブロックは examples からの抜粋なので、
ここに計算を隠すと、紙面に載る抜粋が「どこかで用意された何か」に依存してしまう。
計算はスクリプト本体か ``toukei_tashikame`` パッケージのどちらかにある、の2択にする。
"""

from __future__ import annotations

from toukei_tashikame import plots


def start(title: str = "") -> None:
    """フォントと rcParams を整える。各スクリプトの先頭で1回呼ぶ。"""
    plots.setup()
    if title:
        print(f"--- {title} ---")
