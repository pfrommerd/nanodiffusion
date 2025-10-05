#!/usr/bin/env bash
uv run train --data celeba --wandb --project=nanogen_celeba \
    --model.diffusion.nn.type=unet \
    --optimizer.lr=0.0003 \
    --batch_size 256 \
    --iterations 300_000 \
    --gen_batch_size 64
