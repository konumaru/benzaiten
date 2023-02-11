
.PHONY: generate
generate:
	python src/generate.py \
		demo.name=sample1 \
		exp.name=onehot

.PHONY: tests
tests:
	black src/
	isort src/
	pytest -s --cov=./src

# .PHONY: completion
# completion:
# 	echo eval "$(python src/config.py -sc install=bash)" >> ~/.bashrc
# 	echo [[ $PS1 && -f /usr/share/bash-completion/bash_completion ]] && . /usr/share/bash-completion/bash_completion >> ~/.bashrc
