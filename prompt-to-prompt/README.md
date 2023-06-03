# SalDiff: Saliency Aware Image Generation With Stable Diffusion

## Setup

This code was tested with Python 3.8, [Pytorch](https://pytorch.org/) 1.11

pip install -r requirements.txt

The code was tested on a Tesla V100 16GB but should work on other cards with at least **12GB** VRAM.

## Quickstart

saliency_double_inversion.ipynb contains our approach for guiding human attention. saliency.ipynb includes our preliminary experiments on saliency predictor, prompt-to-prompt and image (dataset) inversion.
