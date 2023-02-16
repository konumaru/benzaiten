
.PHONY: generate
generate:
	python src/generate.py \
		demo.name=sample1,sample2,sample3 \
		exp.name=onehot,embedded

.PHONY: tests
tests:
	black src/
	isort src/
	pytest -s --cov=./src

# .PHONY: completion
# completion:
# 	echo eval "$(python src/config.py -sc install=bash)" >> ~/.bashrc
# 	echo [[ $PS1 && -f /usr/share/bash-completion/bash_completion ]] && . /usr/share/bash-completion/bash_completion >> ~/.bashrc
