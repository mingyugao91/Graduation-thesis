"""配置加载工具"""
import yaml
import copy


def load_config(path="configs/default.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def override_config(config, overrides):
    """递归合并覆盖配置"""
    for key, val in overrides.items():
        if isinstance(val, dict) and key in config:
            override_config(config[key], val)
        else:
            config[key] = val
    return config


def get_ablation_config(base_config, ablation_type):
    """生成消融实验配置

    Args:
        base_config: 基础配置
        ablation_type: "no_aug" | "no_cond" | "no_multiscale" | "full"
    """
    config = copy.deepcopy(base_config)

    if ablation_type == "full":
        config["experiment"]["use_augmentation"] = True
        config["experiment"]["use_conditional"] = True
        config["experiment"]["use_multiscale"] = True
        config["unet"]["cond_dim"] = 4
    elif ablation_type == "no_aug":
        config["experiment"]["use_augmentation"] = False
        config["experiment"]["use_conditional"] = True
        config["experiment"]["use_multiscale"] = True
        config["unet"]["cond_dim"] = 4
    elif ablation_type == "no_cond":
        config["experiment"]["use_augmentation"] = True
        config["experiment"]["use_conditional"] = False
        config["experiment"]["use_multiscale"] = True
        config["unet"]["cond_dim"] = None
    elif ablation_type == "no_multiscale":
        config["experiment"]["use_augmentation"] = True
        config["experiment"]["use_conditional"] = True
        config["experiment"]["use_multiscale"] = False
        config["unet"]["cond_dim"] = 4
    else:
        raise ValueError(f"Unknown ablation type: {ablation_type}")

    return config
