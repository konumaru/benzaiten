
.PHONY: tests



test:
	black src/
	isort src/
	pytest -s --cov=./src
