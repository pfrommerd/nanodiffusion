#!/usr/bin/env bash
uv run train --data trajectory --wandb \
    --model.diffusion.nn.type=unet \
    --batch_size=128 --gen_batch_size=16
