
.PHONY: generate
generate:
	cd docker && docker compose -f docker-compose.cpu.yml up -d
	cd docker && docker compose exec benzaiten-cpu python src/generate.py exp.name=onehot sample_name=sample1

.PHONY: tests
tests:
	black src/
	isort src/
	pytest -s --cov=./src

# .PHONY: completion
# completion:
# 	echo eval "$(python src/config.py -sc install=bash)" >> ~/.bashrc

.PHONY: stop-container
stop-container:
	cd docker && docker compose -f docker-compose.cpu.yml stop