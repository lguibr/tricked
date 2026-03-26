import os

from omegaconf import DictConfig, OmegaConf


def get_hardware_config() -> DictConfig:
    conf_path = os.path.join(os.path.dirname(__file__), "..", "..", "conf", "config.yaml")
    if os.path.exists(conf_path):
        cfg = OmegaConf.load(conf_path)
        assert isinstance(cfg, DictConfig)
        return cfg
    raise FileNotFoundError(
        "Hydra configs dictate conf/config.yaml must explicitly exist; fallbacks deprecated."
    )
