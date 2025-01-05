# Flow Matching in PyTorch

This repository contains a simple PyTorch implementation of the paper [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747).

## 2D Flow Matching Example

The gif below demonstrates mapping a single Gaussian distribution to a checkerboard distribution, with the vector field visualized.

<p align="center">
<img align="middle" src="./outputs/checkerboard/vector_field_checkerboard.gif" height="400" />
</p>

And, here is another example of moons dataset.

<p align="center">
<img align="middle" src="./outputs/moons/vector_field_moons.gif" height="400" />
</p>

## Getting Started

Make sure you have Python 3.10+ installed.

To set up the environment using `uv`:

```bash
uv sync
source .venv/bin/activate
```

Alternatively, using `pip`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Conditional Flow Matching [Lipman+ 2023]

This example demonstrates flow matching on a 2D toy dataset.

```bash
python train_flow_matching_2d.py --dataset checkerboard
```

Several datasets are provided in `flow_matching/datasets.py` such as `checkerboard` and `moons`. The `--dataset` option is required. Training parameters are directly defined in the script, and all visualizations found in `outputs/` are the training results generated with the same default parameters. Also, model checkpoints are not included as they are easily reproducible by running the script above.

## References

- [1] Lipman, Yaron, et al. "Flow Matching for Generative Modeling." [arXiv:2210.02747](https://arxiv.org/abs/2210.02747)
- [2] [facebookresearch/flow_matching](https://github.com/facebookresearch/flow_matching)
- [3] [atong01/conditional-flow-matching](https://github.com/atong01/conditional-flow-matching)
