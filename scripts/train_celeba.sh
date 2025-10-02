#!/usr/bin/env bash
uv run train --data celeba --wandb --project=nanogen_celeba \
    --model.diffusion.nn.type=unet \
    --batch_size 256 \
    --iterations 150_000 \
    --gen_batch_size 64
