#!/usr/bin/env bash
uv run train --data fashion-mnist --wandb --project=nanogen_fashion_mnist \
    --model.diffusion.nn.type=unet \
    --batch_size 256 \
    --iterations 150_000 \
    --gen_batch_size 64
