from dataclasses import dataclass, field

@dataclass
class DiffusionArguments:
    r"""
    Arguments of Diffusion Models.
    """
    diffusion_steps: int = field(
        default=64,
        metadata={"help": "timesteps of diffusion models."}
    )
    decoding_strategy: str = field(
        default="stochastic0.5-linear",
        metadata={"help": "<topk_mode>-<schedule>"}
    )
    topk_decoding: bool = field(
        default=False,
        metadata={"help": "use focal loss for token-level reweighting"}
    )
    time_reweighting: str = field(
        default='original',
        metadata={"help": "use focal loss for token-level reweighting"}
    )
    def __post_init__(self):
        pass
