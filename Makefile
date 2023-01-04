
.PHONY: tests
tests:
	poetry run pytest -s --cov=src/
