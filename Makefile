
.PHONY: generate
generate:
	cd docker && docker compose -f docker-compose.cpu.yml up -d
	cd docker && docker compose exec benzaiten-cpu python src/generate.py exp.name=onehot sample_name=sample1

.PHONY: tests
tests:
	black src/
	isort src/
	pytest -s --cov=./src


.PHONY: dl-soundfont
dl-soundfont:
	mkdir -p ~/data/soundfont
    curl https://github.com/musescore/MuseScore/raw/master/share/sound/FluidR3Mono_GM.sf3 -o ./data/soundfont/FluidR3Mono_GM.sf3
