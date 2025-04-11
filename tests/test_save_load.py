from io import BytesIO

from path_diffusion.datasets.tree_dataset import TreeDataConfig
from path_diffusion.models.mlp import MlpConfig
from path_diffusion.schedules import LogLinearScheduleConfig
from path_diffusion.optimizers import AdamwConfig
from path_diffusion import Diffuser, DiffuserConfig

def test_save_load():
    config = DiffuserConfig(
        schedule=LogLinearScheduleConfig(
            timesteps=1000,
            sigma_min=0.0001,
            sigma_max=1
        ),
        optimizer=AdamwConfig(
            lr=1e-4,
            weight_decay=1e-2,
            betas=(0.9, 0.999),
            eps=1e-8
        ),
        model=MlpConfig(),
        data=TreeDataConfig()
    )
    diffuser = Diffuser.from_config(config)
    
    data = BytesIO()
    diffuser.save(data)
    data.seek(0)
    new_diffuser = Diffuser.load(data)