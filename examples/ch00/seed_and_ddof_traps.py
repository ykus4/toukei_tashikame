"""最初に踏む2つの落とし穴 — シードの流儀と ddof の既定。

どちらも「ライブラリが黙って選んでいる」種類の分岐で、気づかないまま数字が変わる。
本書は前者を default_rng に、後者を ddof=1 に固定する。その理由をここで見ておく。

    uv run python examples/ch00/seed_and_ddof_traps.py
"""

import numpy as np

from toukei_tashikame import describe


def seed_trap() -> None:
    print("--- 落とし穴1: np.random.seed と default_rng ---")

    # 旧い書き方。グローバルな1つの状態を書き換える。
    np.random.seed(0)
    legacy_a = np.random.normal(size=3)
    legacy_b = np.random.normal(size=3)

    # 本書の書き方。rng という「もの」を持ち回る。
    rng = np.random.default_rng(0)
    modern_a = rng.normal(size=3)
    modern_b = rng.normal(size=3)

    print(f"  np.random.seed(0) 1回目   {np.round(legacy_a, 4)}")
    print(f"  np.random.seed(0) 2回目   {np.round(legacy_b, 4)}")
    print(f"  default_rng(0)    1回目   {np.round(modern_a, 4)}")
    print(f"  default_rng(0)    2回目   {np.round(modern_b, 4)}")

    # 問題はここ。グローバル状態は、誰がいつ触ったかを追えない。
    np.random.seed(0)
    _ = np.random.normal(size=100)      # どこかの関数が引いたつもり
    hijacked = np.random.normal(size=3)
    print(f"\n  間に誰かが100個引いたあと  {np.round(hijacked, 4)}")
    print("  ← 同じ seed(0) から始めたのに、値が変わっている。"
          "間に挟まった処理に結果が依存する")

    rng2 = np.random.default_rng(0)
    _ = np.random.default_rng(999).normal(size=100)   # 別の rng が何をしようと
    safe = rng2.normal(size=3)
    print(f"  default_rng を別に持てば   {np.round(safe, 4)}")
    print("  ← 自分の rng だけが自分の列を決める。だから並列にしても再現する")


def ddof_trap() -> None:
    print("\n--- 落とし穴2: ddof の既定が numpy と pandas で違う ---")
    import pandas as pd

    x = np.random.default_rng(0).normal(50.0, 10.0, size=20)
    s = pd.Series(x)

    print(f"  np.std(x)              (ddof=0) {np.std(x):.4f}")
    print(f"  np.std(x, ddof=1)               {np.std(x, ddof=1):.4f}")
    print(f"  pd.Series(x).std()     (ddof=1) {s.std():.4f}")
    print(f"  describe.sd(x)         (ddof=1) {describe.sd(x):.4f}   ← 本書の既定")
    print(f"\n  同じ20個のデータで差は {np.std(x, ddof=1) - np.std(x):.4f}"
          f"（{100 * (np.std(x, ddof=1) / np.std(x) - 1):.1f}%）")
    print("  n が小さいほど開く。numpy と pandas に同じ配列を渡して数字が違うのは、"
          "たいていこれ")

    print("\n  どちらが正しいか、ではない。何を言いたいかで決まる:")
    print("    ddof=0  手元の n 個そのもののばらつき（母集団を持っている）")
    print("    ddof=1  そこから母分散を推す（標本を持っている）  ← 本書はこちら")


def main() -> None:
    seed_trap()
    ddof_trap()


if __name__ == "__main__":
    main()
