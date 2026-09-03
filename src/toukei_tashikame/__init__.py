"""『Pythonで学ぶ！統計学入門』の教材コード。

本文で「道具」として import するのはこのパッケージ。逆に、本文が「手で書く」と宣言した
実装は、その節の ``examples/`` に直書きしてある。ここにあるのは、いったん本文で書いた
あとに何度も使い回すための収納先である。

    from toukei_tashikame import datasets, sim, plots

回して数える手続きは ``sim`` に集約してある。``sim.repeat`` は試行関数に ``rng`` を
渡すので、本書のコードは ``np.random.seed`` を一度も呼ばない。
"""

from __future__ import annotations

__version__ = "0.1.0"

from . import datasets, describe, plots, sim

__all__ = ["__version__", "datasets", "describe", "plots", "sim"]
