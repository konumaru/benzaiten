from dataclasses import dataclass, field

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

hydra_dir = "./data/hydra_output/"
hydra_config = {
    "mode": "MULTIRUN",
    "job": {
        "config": {"override_dirname": {"exclude_keys": ["{seed}"]}},
    },
    "run": {
        "dir": hydra_dir
        + "${exp.name}/${hydra.job.override_dirname}/seed=${seed}"
    },
    "sweep": {
        "dir": hydra_dir
        + "${exp.name}/${hydra.job.override_dirname}/seed=${seed}"
    },
}

# ====================
# Model Config
# ====================


@dataclass
class CVAEModelConfig:
    input_dim: int = 49
    condition_dim: int = 12
    hidden_dim: int = 1024
    latent_dim: int = 256
    num_lstm_layers: int = 2
    num_fc_layers: int = 2
    bidirectional: bool = False


# ====================
# Config
# ====================


# TODO: model nameと必要なパラメータを使って、train.py内でexp_nameを生成する
@dataclass
class ExpConfig:
    name: str = "working"


@dataclass
class Config:
    hydra: DictConfig = OmegaConf.create(hydra_config)

    seed: int = 42
    exp: ExpConfig = field(default_factory=ExpConfig)

    # NTOE: model configs.
    cvae_model: CVAEModelConfig = field(default_factory=CVAEModelConfig)


# NOTE: No further use of cs.store to use auto-completion.
cs: ConfigStore = ConfigStore.instance()
cs.store(name="config", node=Config)


@hydra.main(version_base=None, config_name="config")
def main(cfg: Config) -> None:
    print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    main()
