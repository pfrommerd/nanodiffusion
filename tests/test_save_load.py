from io import BytesIO

from nanodiffusion.datasets.perlin import PerlinDataConfig
from nanodiffusion.models.mlp import MlpConfig
from nanodiffusion.schedules import LogLinearScheduleConfig
from nanodiffusion.optimizers import AdamwConfig
from nanodiffusion.diffuser import Diffuser, DiffuserConfig

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
        data=PerlinDataConfig()
    )
    diffuser = Diffuser.from_config(config)

    data = BytesIO()
    diffuser.save(data)
    data.seek(0)
    new_diffuser = Diffuser.load(data)
