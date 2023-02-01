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


@dataclass(frozen=True)
class Benzaiten:
    root_dir: str = "./data/"
    xml_dir: str = "xml/"
    feature_dir: str = "feature/"

    model_dir: str = "model/"
    train_dir: str = "train/"
    model_filename: str = "best_model.ckpt"

    competition_dir: str = "competition/"
    sample_name: str = "sample1"
    generated_dir: str = "generated/"
    pianoroll_filename: str = "output.png"
    midi_filename: str = "output.midi"
    wav_filename: str = "output.wav"


# ====================
# Feature Config
# ====================


@dataclass(frozen=True)
class Feature:
    # NOTE: FeatureというよりBenzaitenで指定されている値なのでBenzaitenConfigに持たせる
    total_measures: int = 240  # 学習用MusicXMLを読み込む際の小節数の上限
    unit_measures: int = 4  # 1回の生成で扱う旋律の長さ
    beat_reso: int = 4  # 1拍を何個に分割するか（4の場合は16分音符単位）
    n_beats: int = 4  # 1小節の拍数（今回は4/4なので常に4）
    notenum_from: int = 36  # 扱う音域の下限（この値を含む）
    notenum_thru: int = 84  # 扱う音域の上限（この値を含まない）
    intro_blank_measures: int = 4  # ブランクおよび伴奏の小節数の合計
    melody_length: int = 8  # 生成するメロディの長さ（小節数）
    key_root: str = "C"  # 生成するメロディの調のルート（"C" or "A"）
    key_mode: str = "major"  # 生成するメロディの調のモード（"major" or "minor"）
    transpose: int = 12  # 生成するメロディにおける移調量


# ====================
# Model Config
# ====================


@dataclass
class CVAEModelConfig:
    input_dim: int = 49
    condition_dim: int = 12
    hidden_dim: int = 1024
    latent_dim: int = 128
    num_lstm_layers: int = 2
    num_fc_layers: int = 3
    bidirectional: bool = False


# ====================
# Train Config
# ====================


@dataclass
class TrainConfig:
    batch_size: int = 32

    num_epoch: int = 2000
    grad_clip_val: float = 1.0

    learning_rate: float = 3e-4


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

    benzaiten: Benzaiten = field(default_factory=Benzaiten)
    feature: Feature = field(default_factory=Feature)

    # NTOE: model configs.
    cvae_model: CVAEModelConfig = field(default_factory=CVAEModelConfig)
    # vae_model: VAEModelConfig = field(default_factory=VAEModelConfig)

    train: TrainConfig = field(default_factory=TrainConfig)


# NOTE: No further use of cs.store to use auto-completion.
cs: ConfigStore = ConfigStore.instance()
cs.store(name="config", node=Config)


@hydra.main(version_base=None, config_name="config")
def main(cfg: Config) -> None:
    print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    main()
