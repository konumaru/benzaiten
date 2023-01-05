
.PHONY: tests


tests:
	black src/
	isort src/
	pytest -s --cov=./src
