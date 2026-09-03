"""本書で使う numpy はこれだけ。配列・ブロードキャスト・axis・default_rng。

numpy の入門ではなく、**本書を読むのに要る最小セット**を1本で通す。ここに出てこない
機能は、以降の章でも使わない。

    uv run python examples/ch00/numpy_minimum.py
"""

import numpy as np


def show(label: str, value, shape=True) -> None:
    """形と値を並べて出す。numpy の事故はたいてい形の思い違いから来る。"""
    arr = np.asarray(value)
    tag = f"{str(arr.shape):>10}" if shape else " " * 10
    text = np.array2string(arr, precision=4, suppress_small=True, max_line_width=60)
    print(f"{label:<34} {tag}  {text}")


def main() -> None:
    print("--- 1. 配列を作る ---")
    x = np.array([1.0, 2.0, 3.0, 4.0])
    show("np.array([1, 2, 3, 4])", x)
    show("np.arange(5)", np.arange(5))
    show("np.linspace(0, 1, 5)", np.linspace(0, 1, 5))
    show("np.zeros((2, 3))", np.zeros((2, 3)))

    print("\n--- 2. ブロードキャスト（形の違う配列の演算）---")
    col = np.array([[10.0], [20.0], [30.0]])   # (3, 1)
    row = np.array([1.0, 2.0])                 # (2,)
    show("col", col)
    show("row", row)
    show("col + row", col + row)
    print("  (3,1) と (2,) が (3,2) になる。足りない軸は長さ1として補われ、"
          "長さ1の軸は相手に合わせて伸びる")

    print("\n--- 3. axis（どの向きに潰すか）---")
    m = np.arange(6.0).reshape(2, 3)
    show("m", m)
    show("m.sum()          — 全部", m.sum(), shape=False)
    show("m.sum(axis=0)    — 行を潰す", m.sum(axis=0))
    show("m.sum(axis=1)    — 列を潰す", m.sum(axis=1))
    print("  axis は「潰す軸」。axis=0 で潰すと行が消えて、列ごとの合計が残る")

    print("\n--- 4. default_rng（本書はこれしか使わない）---")
    rng = np.random.default_rng(0)
    show("rng.normal(size=4)", rng.normal(size=4))
    show("rng.normal(size=4)  — 続き", rng.normal(size=4))
    again = np.random.default_rng(0)
    show("default_rng(0) を作り直す", again.normal(size=4))
    print("  同じシードから作り直せば同じ列が出る。1つの rng を使い回すと先へ進む")

    print("\n--- 5. 真偽値でのマスク（数え上げの基本形）---")
    sample = np.random.default_rng(1).normal(size=1000)
    over = sample > 1.96
    show("sample > 1.96 の先頭5個", over[:5])
    print(f"  {'割合 = over.mean()':<34} {'':>10}  {over.mean():.4f}"
          "   ← 本書の「数え上げ」はすべてこの形")


if __name__ == "__main__":
    main()
