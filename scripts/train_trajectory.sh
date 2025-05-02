#!/usr/bin/env bash
uv run train --data trajectory --wandb --project=nanogen_trajectory \
    --model.diffusion.nn.type=unet \
    --batch_size=256 --gen_batch_size=16 --iterations 100_000
