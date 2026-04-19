import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


@torch.no_grad()
def encode_dataset(model, data_loader, device, use_mu=True):
    model.eval()

    z_list = []
    mu_list = []
    logvar_list = []
    y_list = []

    for x, y in data_loader:
        x = x.to(device)
        mu, logvar = model.encode(x)
        z = mu if use_mu else model.reparameterize(mu, logvar)

        z_list.append(z.cpu().numpy())
        mu_list.append(mu.cpu().numpy())
        logvar_list.append(logvar.cpu().numpy())
        y_list.append(y.numpy())

    return {
        "z": np.concatenate(z_list, axis=0),
        "mu": np.concatenate(mu_list, axis=0),
        "logvar": np.concatenate(logvar_list, axis=0),
        "y": np.concatenate(y_list, axis=0),
    }


@torch.no_grad()
def interpolate_latent(model, x1, x2, device, steps=10):
    model.eval()

    x1 = x1.to(device)
    x2 = x2.to(device)

    mu1, _ = model.encode(x1)
    mu2, _ = model.encode(x2)

    alphas = torch.linspace(0, 1, steps, device=device)
    zs = [(1 - a) * mu1 + a * mu2 for a in alphas]
    zs = torch.cat(zs, dim=0)

    decoded = model.decode(zs).cpu()
    return decoded


def run_kmeans_on_latent(z, n_clusters=10, random_state=42):
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    pred = model.fit_predict(z)
    return model, pred


def run_gmm_on_latent(z, n_components=10, random_state=42):
    model = GaussianMixture(n_components=n_components, random_state=random_state)
    pred = model.fit_predict(z)
    return model, pred