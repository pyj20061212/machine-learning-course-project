import torch
import torch.nn.functional as F


def vae_loss(recon_x, x, mu, logvar, beta: float = 1.0, recon_loss_type: str = "bce"):
    if recon_loss_type == "bce":
        recon_loss = F.binary_cross_entropy(recon_x, x, reduction="sum")
    elif recon_loss_type == "mse":
        recon_loss = F.mse_loss(recon_x, x, reduction="sum")
    else:
        raise ValueError(f"Unsupported recon_loss_type: {recon_loss_type}")

    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    total_loss = recon_loss + beta * kl_loss

    batch_size = x.size(0)
    return {
        "loss": total_loss / batch_size,
        "recon_loss": recon_loss / batch_size,
        "kl_loss": kl_loss / batch_size,
    }