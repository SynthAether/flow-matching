from pathlib import Path

import matplotlib.animation as animation
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn

from flow_matching.datasets import SyntheticDataset
from flow_matching.solver import ModelWrapper, ODESolver


def save_vector_field_as_gif(
    flow: nn.Module,
    dataset: SyntheticDataset,
    savedir: Path,
    sample_steps: int = 101,
    grid_size: int = 15,
    gif_name: str = "vector_field.gif",
) -> None:
    """
    Save the evolution of colorful vector fields over time as a GIF.

    Args:
        flow: Trained flow model.
        dataset: Dataset object providing the range for the grid.
        savedir: Directory to save the GIF.
        sample_steps: Number of time steps for the animation.
        grid_size: Number of points per axis for the vector field grid.
        gif_name: Name of the output GIF file.
        device: Device for computation ('cpu' or 'cuda').
    """
    flow.eval()
    savedir = Path(savedir)
    savedir.mkdir(parents=True, exist_ok=True)

    # Create a grid for the vector field
    x_range, y_range = dataset.get_square_range()
    x = np.linspace(x_range[0], x_range[1], grid_size)
    y = np.linspace(y_range[0], y_range[1], grid_size)
    xv, yv = np.meshgrid(x, y)
    grid = np.stack([xv, yv], axis=-1).reshape(-1, 2)  # Shape: (grid_size^2, 2)

    device = next(flow.parameters()).device
    grid_tensor = torch.tensor(grid, dtype=torch.float32, device=device)
    time_steps = torch.linspace(0, 1, sample_steps).to(device)

    fig, ax = plt.subplots(figsize=(6, 6))

    def update(frame):
        ax.clear()

        # Compute vector field at time t
        t = time_steps[frame]
        t_tensor = torch.full((grid_tensor.size(0), 1), t, device=device)
        vectors = flow(grid_tensor, t_tensor).detach().cpu().numpy()
        vectors = vectors.reshape(grid_size, grid_size, 2)

        magnitudes = np.linalg.norm(vectors, axis=2)
        norm = plt.Normalize(magnitudes.min(), magnitudes.max())

        # Plot the vector field with colors and scaled arrows
        ax.quiver(
            xv,
            yv,
            vectors[:, :, 0],
            vectors[:, :, 1],
            magnitudes,
            angles="xy",
            scale_units="xy",
            scale=8.0,  # Adjust scale to control arrow length
            cmap=cm.coolwarm,
            norm=norm,
            alpha=0.8,
        )
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        ax.set_title(f"Vector Field (t = {t.item():.2f})", fontsize=14)
        ax.set_aspect("equal")
        ax.axis("off")

    # Create and save the animation as a gif
    ani = animation.FuncAnimation(fig, update, frames=sample_steps, interval=100)
    gif_path = savedir / gif_name
    ani.save(gif_path, writer="pillow")
    print(f"GIF saved to {gif_path}")


def save_vector_field_and_samples_as_gif(
    flow: nn.Module,
    dataset: SyntheticDataset,
    savedir: Path,
    sample_steps: int = 101,
    grid_size: int = 100,
    num_samples: int = 500000,
    gif_name: str = "vector_density_samples.gif",
    interval: int = 50,
):
    """
    Save the evolution of colorful vector fields, density plots, and ODE solver samples as a GIF.

    Args:
        flow: Trained flow model.
        dataset: Dataset object providing the range for the grid.
        savedir: Directory to save the GIF.
        sample_steps: Number of time steps for the animation.
        grid_size: Number of points per axis for the density and vector field grid.
        num_samples: Number of samples to generate with ODE solver.
        gif_name: Name of the output GIF file.
        interval: Interval between frames in milliseconds.
    """
    flow.eval()
    savedir = Path(savedir)
    savedir.mkdir(parents=True, exist_ok=True)

    # Create a grid for the density and vector field
    x_range, y_range = dataset.get_square_range()
    x = np.linspace(x_range[0], x_range[1], grid_size)
    y = np.linspace(y_range[0], y_range[1], grid_size)
    xv, yv = np.meshgrid(x, y)
    grid = np.stack([xv, yv], axis=-1).reshape(-1, 2)  # Shape: (grid_size^2, 2)

    device = next(flow.parameters()).device
    grid_tensor = torch.tensor(grid, dtype=torch.float32, device=device)
    time_steps = torch.linspace(0, 1, sample_steps).to(device)

    # Use ODE solver to sample trajectories
    class WrappedModel(ModelWrapper):
        def forward(self, x: Tensor, t: Tensor, **extras) -> Tensor:
            return self.model(x_t=x, t=t)

    wrapped_model = WrappedModel(flow)
    step_size = 0.05
    x_init = torch.randn((num_samples, 2), dtype=torch.float32, device=device)
    solver = ODESolver(wrapped_model)
    sol = solver.sample(
        x_init=x_init, step_size=step_size, method="midpoint", time_grid=time_steps, return_intermediates=True
    )
    sol = sol.detach().cpu().numpy()

    # Create a figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    def update(frame):
        for ax in axes:
            ax.clear()

        # Current time step
        t = time_steps[frame]
        t_tensor = torch.full((grid_tensor.size(0), 1), t, device=device)

        # Compute vector field at time t
        vectors = flow(grid_tensor, t_tensor).detach().cpu().numpy()
        vectors = vectors.reshape(grid_size, grid_size, 2)
        magnitudes = np.linalg.norm(vectors, axis=2)

        # Normalize for coloring
        norm_vectors = plt.Normalize(magnitudes.min(), magnitudes.max())
        # Flatten magnitudes to 1D array for quiver
        magnitudes_flat = magnitudes.ravel()
        # Define width proportional to vector magnitudes
        width = np.clip(magnitudes_flat / magnitudes_flat.max() * 0.01, 0.002, 0.02)

        # Plot the vector field
        axes[0].quiver(
            xv,
            yv,
            vectors[:, :, 0],
            vectors[:, :, 1],
            magnitudes_flat,
            angles="xy",
            scale_units="xy",
            scale=8.0,  # Adjust scale to control arrow length
            cmap=cm.coolwarm,
            norm=norm_vectors,
            alpha=0.8,
            width=width.mean(),  # Single scalar value or consistent width
        )
        axes[0].set_xlim(x_range)
        axes[0].set_ylim(y_range)
        axes[0].set_title(f"Vector Field (t = {t.item():.2f})", fontsize=16)
        axes[0].set_aspect("equal")
        axes[0].axis("off")

        # Plot the ODE solver samples
        samples = sol[frame]
        axes[1].hist2d(samples[:, 0], samples[:, 1], bins=500, range=[x_range, y_range], cmap="viridis")
        axes[1].set_title(f"Samples (t = {t.item():.2f})", fontsize=16)
        axes[1].set_xlim(x_range)
        axes[1].set_ylim(y_range)
        axes[1].set_aspect("equal")
        axes[1].axis("off")

    # Adjust layout to reduce white space
    # plt.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.05, wspace=0.1)
    fig.tight_layout()

    # Create and save the animation as a GIF
    ani = animation.FuncAnimation(fig, update, frames=sample_steps, interval=interval)
    path = savedir / gif_name
    ani.save(path, writer="pillow")
    print(f"Animation saved to {path}")


def plot_likelihood(
    flow: nn.Module,
    dataset: SyntheticDataset,
    output_dir: str = "./",
    filename: str = "likelihood.png",
    device: str = "cpu",
):
    # Adapted from https://github.com/facebookresearch/flow_matching/blob/main/examples/2d_flow_matching.ipynb

    from torch.distributions import Independent, Normal

    flow.eval()

    # sample with likelihood
    step_size = 0.05

    square_range = dataset.get_square_range()
    extent = sum(square_range, [])  # flatten
    grid_size = 200
    x_1 = torch.meshgrid(
        torch.linspace(extent[1], extent[0], grid_size),
        torch.linspace(extent[2], extent[3], grid_size),
        indexing="ij",
    )
    x_1 = torch.stack([x_1[0].flatten(), x_1[1].flatten()], dim=1).to(device)

    # source distribution is an isotropic gaussian
    gaussian_log_density = Independent(Normal(torch.zeros(2, device=device), torch.ones(2, device=device)), 1).log_prob

    # Use ODE solver to sample trajectories
    class WrappedModel(ModelWrapper):
        def forward(self, x: Tensor, t: Tensor, **extras) -> Tensor:
            return self.model(x_t=x, t=t)

    wrapped_model = WrappedModel(flow)
    solver = ODESolver(wrapped_model)

    # compute log likelihood with unbiased hutchinson estimator, average over num_acc
    num_acc = 10
    log_p_acc = 0

    for _ in range(num_acc):
        _, log_p = solver.compute_likelihood(
            x_1=x_1, method="midpoint", step_size=step_size, exact_divergence=False, log_p0=gaussian_log_density
        )
        log_p_acc += log_p
    log_p_acc /= num_acc

    # compute with exact divergence
    sol, exact_log_p = solver.compute_likelihood(
        x_1=x_1,
        method="midpoint",
        step_size=step_size,
        exact_divergence=True,
        log_p0=gaussian_log_density,
        return_intermediates=True,
    )
    sol = sol.detach().cpu().numpy()

    # plot
    likelihood = torch.exp(log_p_acc).cpu().reshape(grid_size, grid_size).detach().numpy().T
    exact_likelihood = torch.exp(exact_log_p).cpu().reshape(grid_size, grid_size).detach().numpy().T

    fig, axs = plt.subplots(1, 2, figsize=(10, 6))

    cmin = 0.0
    cmax = 1 / 32  # 1/32 is the gt likelihood value

    norm = cm.colors.Normalize(vmax=cmax, vmin=cmin)

    axs[0].imshow(likelihood, extent=extent, origin="upper", cmap="viridis", norm=norm)
    axs[0].set_title(f"Model Likelihood, Hutchinson Estimator, #acc={num_acc}")
    axs[1].imshow(exact_likelihood, extent=extent, origin="upper", cmap="viridis", norm=norm)
    axs[1].set_title("Exact Model Likelihood")

    fig.colorbar(cm.ScalarMappable(norm=norm, cmap="viridis"), ax=axs, orientation="horizontal", label="density")

    plt.savefig(Path(output_dir) / filename)
    print("Likelihood saved to", Path(output_dir) / filename)
