.PHONY: test figures run-all run-all-slow lint sync

sync:
	uv sync --extra dev

test:
	uv run pytest -q

# @slow（PyMC）を含む全部
test-all:
	uv sync --extra dev --extra slow && uv run pytest -q -m ""

lint:
	uv run ruff check src examples tests

# 全 examples を通しで実行する。CI が毎回やることと同じ。
#
# PyMC を要するスクリプトは docstring の先頭に `@slow` と書いてある。PyMC は
# 既定の依存に入っていない（uv sync --extra slow が要る）ので、入っていない環境では
# 飛ばす。ここで飛ばさないと、PyMC のビルドを待たせないための切り分けが CI の
# 失敗として跳ね返ってくる。@slow を全部走らせたいときは make run-all-slow。
HAS_PYMC := $(shell uv run python -c "import importlib.util,sys; sys.stdout.write('1' if importlib.util.find_spec('pymc') else '')" 2>/dev/null)

run-all:
	@set -e; skipped=0; \
	for f in $$(find examples -name '*.py' -not -name '_*' | sort); do \
		if grep -q '@slow' $$f && [ -z "$(HAS_PYMC)" ]; then \
			echo "--- $$f  (skip: PyMC 未導入)"; skipped=$$((skipped+1)); continue; \
		fi; \
		echo "--- $$f"; uv run python $$f > /dev/null; \
	done; \
	echo "全 examples が完走した（skip $$skipped 本）"

run-all-slow:
	@uv sync --extra dev --extra slow
	@$(MAKE) run-all

figures: run-all
