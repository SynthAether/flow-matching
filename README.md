# Flow Matching in PyTorch

This repository contains a simple PyTorch implementation of the paper [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747).

## 2D Flow Matching Example

The gif below demonstrates mapping a single Gaussian distribution to a checkerboard distribution, with the vector field visualized.

<p align="center">
<img align="middle" src="./outputs/cfm/checkerboard/vector_field_checkerboard.gif" height="400" />
</p>

And, here is another example of moons dataset.

<p align="center">
<img align="middle" src="./outputs/cfm/moons/vector_field_moons.gif" height="400" />
</p>

## Getting Started

Clone the repository and set up the python environment.

```bash
git clone https://github.com/keishihara/flow-matching.git
cd flow-matching
```

Make sure you have Python 3.10+ installed.
To set up the python environment using `uv`:

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

This is the original CFM paper implementation [1]. Some components of the code are adapted from [2] and [3].

### 2D Toy Datasets

You can train the CFM models on 2D synthetic datasets such as `checkerboard` and `moons`. Specify the dataset name using `--dataset` option. Training parameters are predefined in the script, and visualizations of the training results are stored in the `outputs/` directory. Model checkpoints are not included as they are easily reproducible with the default settings.

```bash
python train_flow_matching_2d.py --dataset checkerboard
```

The vector fields and generated samples, like the ones displayed as GIFs at the top of this README, can now be found in the `outputs/cfm/` directory.

### Image Datasets

You can also train class-conditional CFM models on popular image classification datasets. Both the generated samples and model checkpoints will be stored in the `outputs/cfm` directory. For a detailed list of training parameters, run `python train_flow_matching_on_images.py --help`.

To train a class-conditional CFM on MNIST dataset, run:

```bash
python train_flow_matching_on_images.py --do_train --dataset mnist
```

After training, you can now generate samples with:

```bash
python train_flow_matching_on_images.py --do_sample --dataset mnist
```

Now, you should be able to see the generated samples in the `outputs/cfm/mnist/` directory.

<p align="center">
<img align="middle" src="./outputs/cfm/mnist/trajectory.gif" height="400" />
</p>

## References

- [1] Lipman, Yaron, et al. "Flow Matching for Generative Modeling." [arXiv:2210.02747](https://arxiv.org/abs/2210.02747)
- [2] [facebookresearch/flow_matching](https://github.com/facebookresearch/flow_matching)
- [3] [atong01/conditional-flow-matching](https://github.com/atong01/conditional-flow-matching)
