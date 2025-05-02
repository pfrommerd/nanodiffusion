#!/usr/bin/env bash
uv run train --data mnist --wandb \
    --model.diffusion.nn.type=unet \
    --batch_size 128 \
    --iterations 150_000 \
    --gen_batch_size 64 \
