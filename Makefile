
.PHONY: generate
generate:
	python src/generate.py \
	demo.name=sample1 \
	exp.name=LSTM,LSTM_hiddenDim-256,LSTM_hiddenDim-2048

.PHONY: tests
tests:
	black src/
	isort src/
	pytest -s --cov=./src
