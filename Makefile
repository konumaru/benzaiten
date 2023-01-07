
.PHONY: generate
generate:
	python src/generate.py \
	demo.name=sample2 \
	exp.name=LSTM,LSTM_hiddenDim-256

.PHONY: tests
tests:
	black src/
	isort src/
	pytest -s --cov=./src
