#!/usr/bin/env bash
uv run train --data celeba --wandb --project=nanogen_celeba \
    --model.diffusion.nn.type=unet \
    --model.diffusion.nn.unet.base_channels=128 \
    --optimizer.lr=0.0002 \
    --checkpoint_interval=10_000 \
    --batch_size 256 \
    --iterations 150_000 \
    --gen_batch_size 64
