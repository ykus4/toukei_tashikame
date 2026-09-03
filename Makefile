.PHONY: test figures run-all lint sync

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
run-all:
	@set -e; for f in $$(find examples -name '*.py' -not -name '_*' | sort); do \
		echo "--- $$f"; uv run python $$f > /dev/null; \
	done; echo "全 examples が完走した"

figures: run-all
