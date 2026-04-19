import numpy as np
import matplotlib.pyplot as plt
import torch


def plot_training_curves(history, save_path=None):
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="train total")
    plt.plot(epochs, history["val_loss"], label="val total")
    plt.plot(epochs, history["train_recon_loss"], label="train recon")
    plt.plot(epochs, history["val_recon_loss"], label="val recon")
    plt.plot(epochs, history["train_kl_loss"], label="train kl")
    plt.plot(epochs, history["val_kl_loss"], label="val kl")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("VAE Training Curves")
    plt.legend()
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()


def show_reconstructions(originals, reconstructions, n=8, save_path=None):
    n = min(n, len(originals))
    fig, axes = plt.subplots(2, n, figsize=(1.8 * n, 4))

    for i in range(n):
        axes[0, i].imshow(originals[i].squeeze(), cmap="gray")
        axes[0, i].axis("off")
        axes[0, i].set_title("Orig")

        axes[1, i].imshow(reconstructions[i].squeeze(), cmap="gray")
        axes[1, i].axis("off")
        axes[1, i].set_title("Recon")

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()


def show_generated_samples(samples, n=16, save_path=None):
    n = min(n, len(samples))
    side = int(np.ceil(np.sqrt(n)))
    fig, axes = plt.subplots(side, side, figsize=(2 * side, 2 * side))
    axes = np.array(axes).reshape(-1)

    for i in range(len(axes)):
        axes[i].axis("off")
        if i < n:
            axes[i].imshow(samples[i].squeeze(), cmap="gray")

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()


def plot_latent_scatter(z, y, save_path=None):
    plt.figure(figsize=(7, 6))
    scatter = plt.scatter(z[:, 0], z[:, 1], c=y, s=8, alpha=0.7)
    plt.colorbar(scatter)
    plt.xlabel("z1")
    plt.ylabel("z2")
    plt.title("2D Latent Space")
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()



def show_multi_method_reconstructions(originals, reconstructions_dict, labels=None, n=8, save_path=None):
    import numpy as np
    import matplotlib.pyplot as plt

    n = min(n, len(originals))
    method_names = list(reconstructions_dict.keys())
    n_rows = 1 + len(method_names)

    fig, axes = plt.subplots(n_rows, n, figsize=(1.8 * n, 1.8 * n_rows))

    if n_rows == 1:
        axes = np.array([axes])

    for i in range(n):
        axes[0, i].imshow(originals[i].squeeze(), cmap="gray")
        axes[0, i].axis("off")
        if labels is not None:
            axes[0, i].set_title(str(labels[i]))

    axes[0, 0].set_ylabel("Orig", rotation=90)

    for row, method in enumerate(method_names, start=1):
        recons = reconstructions_dict[method]
        for i in range(n):
            axes[row, i].imshow(recons[i].squeeze(), cmap="gray")
            axes[row, i].axis("off")
        axes[row, 0].set_ylabel(method, rotation=90)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()


@torch.no_grad()
def plot_latent_grid(model, device, grid_size=20, latent_range=(-3, 3), save_path=None):
    if model.latent_dim != 2:
        raise ValueError("plot_latent_grid requires latent_dim == 2")

    grid_x = np.linspace(latent_range[0], latent_range[1], grid_size)
    grid_y = np.linspace(latent_range[0], latent_range[1], grid_size)

    canvas = np.zeros((28 * grid_size, 28 * grid_size))

    for i, yi in enumerate(grid_y[::-1]):
        for j, xi in enumerate(grid_x):
            z = torch.tensor([[xi, yi]], dtype=torch.float32, device=device)
            x_recon = model.decode(z).cpu().numpy()[0, 0]
            canvas[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = x_recon

    plt.figure(figsize=(10, 10))
    plt.imshow(canvas, cmap="gray")
    plt.axis("off")
    plt.title("Latent Grid Sampling")
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()