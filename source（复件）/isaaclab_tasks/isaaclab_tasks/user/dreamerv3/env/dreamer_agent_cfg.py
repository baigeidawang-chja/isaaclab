from isaaclab.utils.configclass import configclass


@configclass
class DreamerAgentCfg:
    """占位 + 可通过 Hydra CLI 覆盖的 Dreamer agent 配置。
    注意：这里放的应是 Dreamer 训练循环希望 Hydration 的那部分参数。
    """
    # 例子：你可以按需加字段
    batch_size: int = 4
    batch_length: int = 16
    imag_length: int = 15
    horizon: int = 333
    lr: float = 4e-5