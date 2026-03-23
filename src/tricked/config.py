import os

from omegaconf import DictConfig, OmegaConf


def get_hardware_config() -> DictConfig:
    conf_path = os.path.join(os.path.dirname(__file__), "..", "..", "conf", "config.yaml")
    if os.path.exists(conf_path):
        cfg = OmegaConf.load(conf_path)
    else:
        cfg = OmegaConf.create({
            "device": "cuda",
            "model_checkpoint": "runs/default/model.pth",
            "metrics_file": "runs/default/metrics.json",
            "d_model": 128,
            "num_blocks": 8,
            "capacity": 200000,
            "num_games": 1024,
            "simulations": 128,        
            "train_batch_size": 1024,
            "train_epochs": 4,         
            "num_processes": 24,       
            "worker_device": "cpu",
            "unroll_steps": 5,
            "td_steps": 10,
            "zmq_inference_port": "tcp://127.0.0.1:5555",
            "zmq_batch_size": 24,      
            "zmq_timeout_ms": 2,       
            "max_gumbel_k": 8,
            "gumbel_scale": 1.0,
            "temp_decay_steps": 30,
            "difficulty": 6,
            "exploit_starts": [],
            "temp_boost": False,
            "exp_name": "Headless-CUDA-Training",
            "lr_init": 1e-3
        })
    assert isinstance(cfg, DictConfig)
    return cfg
