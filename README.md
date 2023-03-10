# Benzaiten

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)

Here the chord2melody model is implemented in pytorch.

## Generate

```sh
(venv)$ python src/generate.py exp.name=onehot,embedded sample_name=sample1
```

## Model Overview

### input data

[Charlie Parker's Omnibook MusicXML data](https://homepages.loria.fr/evincent/omnibook/)

### model

<img src="data/tmp/model.png">
