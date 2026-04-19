import torch
from .losses import vae_loss


@torch.no_grad()
def evaluate_vae(model, data_loader, device, beta=1.0, recon_loss_type="bce"):
    model.eval()

    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    n_samples = 0

    for x, _ in data_loader:
        x = x.to(device)
        recon_x, mu, logvar = model(x)
        loss_dict = vae_loss(recon_x, x, mu, logvar, beta=beta, recon_loss_type=recon_loss_type)

        batch_size = x.size(0)
        total_loss += loss_dict["loss"].item() * batch_size
        total_recon += loss_dict["recon_loss"].item() * batch_size
        total_kl += loss_dict["kl_loss"].item() * batch_size
        n_samples += batch_size

    return {
        "loss": total_loss / n_samples,
        "recon_loss": total_recon / n_samples,
        "kl_loss": total_kl / n_samples,
    }


@torch.no_grad()
def reconstruct_images(model, x, device):
    model.eval()
    x = x.to(device)
    recon_x, mu, logvar = model(x)
    return recon_x.cpu(), mu.cpu(), logvar.cpu()


@torch.no_grad()
def sample_from_prior(model, n_samples, device):
    model.eval()
    z = torch.randn(n_samples, model.latent_dim).to(device)
    samples = model.decode(z)
    return samples.cpu()