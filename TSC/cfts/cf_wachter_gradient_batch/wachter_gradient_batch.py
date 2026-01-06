import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np



def euclidean_dist(x, y):
    # x, y: (B, C, L)
    return torch.norm((x - y).reshape(x.shape[0], -1), dim=1)

def manhattan_dist(x, y):
    return torch.sum(torch.abs(x - y).reshape(x.shape[0], -1), dim=1)


def cf_ts(
    samples,        # (B, C, L)
    dataset,
    model,
    targets=None,   # None or (B,)
    lb=None,
    lb_step=None,
    max_cfs=1000,
    full_random=False,
    distance="euclidean",
    verbose=False
):
    device = next(model.parameters()).device
    model.eval()

    # -----------------------------
    # Input normalization
    # -----------------------------
    samples = torch.tensor(samples, dtype=torch.float32, device=device)

    if samples.ndim == 2:
        samples = samples.unsqueeze(1)  # (B, 1, L)

    B, C, L = samples.shape
    sample_t = samples.clone()

    # -----------------------------
    # Initial predictions
    # -----------------------------
    with torch.no_grad():
        logits = model(sample_t)
        probs = torch.softmax(logits, dim=1)
        labels = probs.argmax(dim=1)

    if targets is None:
        targets = torch.argsort(probs, dim=1)[:, -2]
    else:
        targets = torch.tensor(targets, device=device, dtype=torch.long)

    # -----------------------------
    # Initialize CFs
    # -----------------------------
    if full_random:
        sample_cf = sample_t + 0.1 * torch.randn_like(sample_t)
    else:
        idxs = torch.randint(0, len(dataset), (B,))
        x0 = []
        for i in idxs:
            xi = torch.tensor(dataset[int(i)][0], device=device, dtype=torch.float32)
            if xi.ndim == 1:
                xi = xi.unsqueeze(0)
            x0.append(xi)
        sample_cf = torch.stack(x0, dim=0)

    sample_cf.requires_grad_(True)

    # -----------------------------
    # Distance fn
    # -----------------------------
    dist_fn = euclidean_dist if distance == "euclidean" else manhattan_dist

    # -----------------------------
    # λ initialization
    # -----------------------------
    with torch.no_grad():
        d0 = dist_fn(sample_t, sample_cf)

    if lb is None:
        lb = torch.clamp(d0 / 10.0, min=0.1)
    else:
        lb = torch.full((B,), float(lb), device=device)

    if lb_step is None:
        lb_step = torch.clamp(d0 / 100.0, min=0.01)
    else:
        lb_step = torch.full((B,), float(lb_step), device=device)

    # -----------------------------
    # Optimizer
    # -----------------------------
    optimizer = Adam([sample_cf], lr=1e-3)
    ce = nn.CrossEntropyLoss(reduction="none")

    # Mask: 1 = still optimizing, 0 = frozen
    active = torch.ones(B, device=device)

    best_validity = torch.zeros(B, device=device)

    # -----------------------------
    # Optimization loop
    # -----------------------------
    for it in range(max_cfs):
        optimizer.zero_grad()

        logits = model(sample_cf)
        cls_loss = ce(logits, targets)          # (B,)
        dist = dist_fn(sample_t, sample_cf)     # (B,)

        loss_per_sample = lb * (cls_loss ** 2) + dist
        loss = (loss_per_sample * active).mean()
        loss.backward()

        # Zero gradients for inactive samples
        sample_cf.grad *= active.view(B, 1, 1)

        optimizer.step()

        # -------------------------
        # Update λ
        # -------------------------
        if it % 10 == 0:
            lb = lb + lb_step * active

        # -------------------------
        # Check validity
        # -------------------------
        with torch.no_grad():
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)
            validity = probs[torch.arange(B), targets]

            improved = validity > best_validity
            best_validity[improved] = validity[improved]

            newly_done = (preds == targets) & (active == 1)
            active[newly_done] = 0

        if verbose and it % 200 == 0:
            done = int((active == 0).sum().item())
            print(f"iter {it}: done {done}/{B}")

        if active.sum() == 0:
            if verbose:
                print(f"All CFs found at iter {it}")
            break

    # -----------------------------
    # Output
    # -----------------------------
    with torch.no_grad():
        final_logits = model(sample_cf)
        final_probs = torch.softmax(final_logits, dim=1)

    return (
        sample_cf,#.detach().cpu().numpy(),   # (B, C, L)
        final_probs#.detach().cpu().numpy()  # (B, num_classes)
    )
