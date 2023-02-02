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
    root_dir: str = "/workspace/data"
    xml_dir: str = "xml/"
    model_dir: str = "model/"
    feat_dir: str = "feats/"


@dataclass(frozen=True)
class Feature:
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


@dataclass
class Preprocessing:
    xml_url: str = (
        "https://homepages.loria.fr/evincent/omnibook/omnibook_xml.zip"
    )
    feat_file: str = "benzaiten_feats.pkl"


# ====================
# Model Config
# ====================


@dataclass
class Encoder:
    input_dim: int = 61
    emb_dim: int = 1024
    hidden_dim: int = 1024
    n_layers: int = 1


@dataclass
class Decoder:
    output_dim: int = 49
    hidden_dim: int = 1024
    n_layers: int = 1


@dataclass
class VAE:
    hidden_dim: int = 1024
    latent_dim: int = 32
    n_hidden: int = 0  # number of hidden-to-hidden layers -> スターターキットは0でOK
    kl_weight: float = 0.001  # VAEにおけるKL項（正規化項）にかける重み


@dataclass
class LSTMConfig:
    name: str = "LSTM"
    hidden_dim: int = 1024

    encoder: Encoder = field(default=Encoder(hidden_dim=hidden_dim))
    decoder: Decoder = field(default=Decoder(hidden_dim=hidden_dim))
    vae: VAE = field(default=VAE())


# TODO: implement music transformer.
@dataclass
class MusicTransformerConfig:
    name: str = "MusicTransfomer"


@dataclass
class OptimizerConfig:
    name: str = "Adam"
    lr: float = 3e-4
    params: DictConfig = DictConfig(
        {
            "lr": lr,
            "betas": [0.9, 0.999],
            "eps": 1e-08,
            "weight_decay": 0,
        }
    )


@dataclass
class SchedulerConfig:
    name: str = "MultiStepLR"
    params: DictConfig = DictConfig({"milestones": [1000, 1500], "gamma": 0.6})


# ====================
# Train Config
# ====================


@dataclass
class TrainConfig:
    n_epoch: int = 3000
    n_batch: int = 32

    optimizer: OptimizerConfig = field(default=OptimizerConfig())
    lr_scheduler: SchedulerConfig = field(default=SchedulerConfig())

    use_grad_clip: bool = False  # 勾配クリッピングを使うか否か
    grad_max_norm: float = 1.0  # 勾配クリッピングのしきい値
    use_scheduler: bool = True  # 学習率スケジューリングを使うか否か


# ====================
# Demo Config
# ====================


@dataclass
class DemoConfig:

    name: str = "sample1"
    input_dir: str = "comp_inputs/"

    output_dir: str = "generated/"
    wav_dir: str = "all_wavs/"


@dataclass
class ExpConfig:
    name: str = "working"


@dataclass
class Config:
    exp: ExpConfig = field(default_factory=ExpConfig)
    seed: int = 42
    benzaiten: Benzaiten = field(default_factory=Benzaiten)
    preprocess: Preprocessing = field(default_factory=Preprocessing)
    feature: Feature = field(default_factory=Feature)
    model: LSTMConfig = field(default_factory=LSTMConfig)
    training: TrainConfig = field(default=TrainConfig())
    demo: DemoConfig = field(default_factory=DemoConfig)

    hydra: DictConfig = OmegaConf.create(hydra_config)


# NOTE: No further use of cs.store to use auto-completion.
cs: ConfigStore = ConfigStore.instance()
cs.store(name="config", node=Config)


@hydra.main(version_base=None, config_name="config")
def main(cfg: Config) -> None:
    print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    main()
