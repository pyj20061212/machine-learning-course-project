import copy
import torch
from .losses import vae_loss


def train_one_epoch(model, data_loader, optimizer, device, beta=1.0, recon_loss_type="bce"):
    model.train()

    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    n_samples = 0

    for x, _ in data_loader:
        x = x.to(device, non_blocking=True)

        optimizer.zero_grad()
        recon_x, mu, logvar = model(x)
        loss_dict = vae_loss(recon_x, x, mu, logvar, beta=beta, recon_loss_type=recon_loss_type)
        loss = loss_dict["loss"]
        loss.backward()
        optimizer.step()

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
def validate_one_epoch(model, data_loader, device, beta=1.0, recon_loss_type="bce"):
    model.eval()

    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    n_samples = 0

    for x, _ in data_loader:
        x = x.to(device, non_blocking=True)
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


def get_beta(epoch: int, warmup_epochs: int, target_beta: float = 1.0):
    return min(target_beta, target_beta * epoch / warmup_epochs)


def train_vae(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    epochs=20,
    beta=1.0,
    recon_loss_type="bce",
    save_path=None,
    warmup_epochs=None,
):
    history = {
        "beta": [],
        "train_loss": [],
        "train_recon_loss": [],
        "train_kl_loss": [],
        "val_loss": [],
        "val_recon_loss": [],
        "val_kl_loss": [],
    }

    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(1, epochs + 1):
        if warmup_epochs is None or warmup_epochs <= 0:
            current_beta = beta
        else:
            current_beta = get_beta(epoch, warmup_epochs=warmup_epochs, target_beta=beta)

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, device,
            beta=current_beta, recon_loss_type=recon_loss_type
        )
        val_metrics = validate_one_epoch(
            model, val_loader, device,
            beta=current_beta, recon_loss_type=recon_loss_type
        )

        history["beta"].append(current_beta)
        history["train_loss"].append(train_metrics["loss"])
        history["train_recon_loss"].append(train_metrics["recon_loss"])
        history["train_kl_loss"].append(train_metrics["kl_loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_recon_loss"].append(val_metrics["recon_loss"])
        history["val_kl_loss"].append(val_metrics["kl_loss"])

        print(
            f"Epoch [{epoch:02d}/{epochs}] "
            f"beta={current_beta:.3f} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_recon={train_metrics['recon_loss']:.4f} "
            f"train_kl={train_metrics['kl_loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_recon={val_metrics['recon_loss']:.4f} "
            f"val_kl={val_metrics['kl_loss']:.4f}"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_state = copy.deepcopy(model.state_dict())
            if save_path is not None:
                torch.save(best_state, save_path)

    model.load_state_dict(best_state)
    return history