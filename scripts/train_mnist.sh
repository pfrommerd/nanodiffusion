#!/usr/bin/env bash
uv run train --data mnist --wandb --project=nanogen_mnist \
    --model.diffusion.nn.type=unet \
    --batch_size 256 \
    --iterations 150_000 \
    --gen_batch_size 64 \
    --gamma 1.0 \
    --mu 0.5 \
    --distill
uv run train --data mnist --wandb --project=nanogen_mnist \
    --model.diffusion.nn.type=unet \
    --batch_size 256 \
    --iterations 150_000 \
    --gen_batch_size 64 \
    --gamma 1.0 \
    --mu 0. \
    --distill
