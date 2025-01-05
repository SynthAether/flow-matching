import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import cm
from scipy.ndimage import gaussian_filter1d
from torch import Tensor, nn
from torch.nn import Module

from flow_matching import visualization
from flow_matching.datasets import TOY_DATASETS
from flow_matching.solver import ModelWrapper, ODESolver


class Swish(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x * torch.sigmoid(x)


class Mlp(Module):
    def __init__(self, dim: int = 2, time_dim: int = 1, h: int = 64) -> None:
        super().__init__()
        self.input_dim = dim
        self.time_dim = time_dim
        self.hidden_dim = h
        self.layers = nn.Sequential(
            nn.Linear(dim + time_dim, h),
            Swish(),
            nn.Linear(h, h),
            Swish(),
            nn.Linear(h, h),
            Swish(),
            nn.Linear(h, dim),
        )

    def forward(self, x_t: Tensor, t: Tensor) -> Tensor:
        size = x_t.size()
        x_t = x_t.reshape(-1, self.input_dim)
        t = t.reshape(-1, self.time_dim).float()
        t = t.reshape(-1, 1).expand(x_t.size(0), 1)
        h = torch.cat([x_t, t], dim=1)
        output = self.layers(h)
        return output.reshape(*size)

    def step(self, x_t: Tensor, t_start: Tensor, t_end: Tensor) -> Tensor:
        # This will be used to sample from the model
        t_start = t_start.unsqueeze(-1).expand(x_t.size(0), 1)
        t_mid = t_start + (t_end - t_start) / 2
        x_mid = x_t + self(x_t=x_t, t=t_start) * (t_end - t_start) / 2
        flow_mid = self(x_t=x_mid, t=t_mid)
        return x_t + (t_end - t_start) * flow_mid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=TOY_DATASETS.keys(), required=True)
    parser.add_argument("--output-dir", type=str, default="outputs")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    args.output_dir = Path(args.output_dir) / args.dataset
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {device}")
    print(f"Dataset: {args.dataset}")

    # Training parameters
    learning_rate = 1e-3
    batch_size = 4096
    iterations = 20000
    log_every = 2000
    hidden_dim = 512

    dataset = TOY_DATASETS[args.dataset](device=device)

    flow = Mlp(dim=dataset.dim, time_dim=1, h=hidden_dim).to(device)
    optimizer = torch.optim.AdamW(flow.parameters(), learning_rate)

    # Training

    losses = []
    for global_step in range(iterations):
        x_1 = dataset.sample(batch_size)
        x_0 = torch.randn_like(x_1).to(device)
        t = torch.rand(x_1.size(0), 1).to(device)

        # Compute the Conditional Flow Matching objective
        # Check eq. (22) and (23) in the paper: https://arxiv.org/abs/2210.02747
        # where, we set \sigma_{\min} = 0
        x_t = (1 - t) * x_0 + t * x_1  # \phi_t(x_0)
        dx_t = x_1 - x_0  # u_t(x|x_t)

        optimizer.zero_grad()
        loss = F.mse_loss(flow(x_t=x_t, t=t), dx_t)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if (global_step + 1) % log_every == 0:
            print(f"| step: {global_step+1:6d} | loss: {loss.item():8.4f} |")

    flow.eval()

    # Plot learning curves

    steps = np.arange(1, len(losses) + 1)
    smoothed_losses = gaussian_filter1d(losses, sigma=5)
    blue = "#1f77b4"
    plt.figure(figsize=(6, 5))
    plt.plot(steps, losses, color=blue, alpha=0.3)
    plt.plot(steps, smoothed_losses, color=blue, linewidth=2)
    plt.title("Training dynamics", fontsize=16)
    plt.xlabel("Steps", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(Path(args.output_dir) / "losses.png")
    print("Training curves saved to", Path(args.output_dir) / "losses.png")

    # Sampling (naive)

    x = torch.randn(batch_size, 2).to(device)
    n_steps = 8
    fig, axes = plt.subplots(1, n_steps + 1, figsize=(30, 4), sharex=True, sharey=True)
    time_steps = torch.linspace(0, 1.0, n_steps + 1).to(device)

    axes[0].scatter(x.detach().cpu()[:, 0], x.detach().cpu()[:, 1], s=10)
    axes[0].set_title(f"t = {time_steps[0]:.2f}")
    axes[0].set_xlim(-3.0, 3.0)
    axes[0].set_ylim(-3.0, 3.0)
    for i in range(n_steps):
        x = flow.step(x_t=x, t_start=time_steps[i], t_end=time_steps[i + 1])
        axes[i + 1].scatter(x.detach().cpu()[:, 0], x.detach().cpu()[:, 1], s=10)
        axes[i + 1].set_title(f"t = {time_steps[i + 1]:.2f}")
    plt.tight_layout()
    plt.savefig(Path(args.output_dir) / f"sampling_{args.dataset}.png")
    print("Sampling results saved to", Path(args.output_dir) / f"sampling_{args.dataset}.png")

    # Sampling with ODE solver

    class WrappedModel(ModelWrapper):
        def forward(self, x: Tensor, t: Tensor, **extras) -> Tensor:
            return self.model(x_t=x, t=t)

    wrapped_model = WrappedModel(flow)

    step_size = 0.05
    norm = cm.colors.Normalize(vmax=50, vmin=0)
    num_samples = 1000000
    sample_steps = 10
    T = torch.linspace(0, 1, sample_steps)  # sample times
    T = T.to(device)

    x_init = torch.randn((num_samples, 2), dtype=torch.float32, device=device)
    solver = ODESolver(wrapped_model)
    sol = solver.sample(x_init=x_init, step_size=step_size, method="midpoint", time_grid=T, return_intermediates=True)
    sol = sol.detach().cpu().numpy()
    T = T.cpu()

    square_range = dataset.get_square_range(samples=torch.from_numpy(sol[-1]))
    fig, axes = plt.subplots(1, sample_steps, figsize=(2 * sample_steps, 2))

    for i in range(sample_steps):
        H = axes[i].hist2d(sol[i, :, 0], sol[i, :, 1], bins=300, norm=norm, range=square_range)
        cmin = 0.0
        cmax = torch.quantile(torch.from_numpy(H[0]), 0.99).item()
        norm = cm.colors.Normalize(vmax=cmax, vmin=cmin)
        axes[i].hist2d(sol[i, :, 0], sol[i, :, 1], bins=300, norm=norm, range=square_range)
        axes[i].set_aspect("equal")
        axes[i].axis("off")
        axes[i].set_title(f"t = {T[i]:.2f}")

    plt.tight_layout()
    plt.savefig(Path(args.output_dir) / f"sampling_{args.dataset}_w_solver.png")
    print("Sampling results with ODE solver saved to", Path(args.output_dir) / f"sampling_{args.dataset}_w_solver.png")

    # Visualize the vector field and samples
    visualization.save_vector_field_and_samples_as_gif(
        flow=flow,
        dataset=dataset,
        savedir=args.output_dir,
        sample_steps=101,
        grid_size=15,
        gif_name=f"vector_field_{args.dataset}.gif",
        interval=50,
    )


if __name__ == "__main__":
    main()
