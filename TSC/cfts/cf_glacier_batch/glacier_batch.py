import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Optional, Tuple


def cf_ts(
    samples: np.ndarray,                   # (B, C, L)
    dataset,
    model: nn.Module,
    targets: Optional[np.ndarray] = None,   # (B,)
    lambda_sparse: float = 0.1,
    lambda_proximity: float = 1.0,
    lambda_diversity: float = 0.1,
    max_iterations: int = 1000,
    learning_rate: float = 0.01,
    tolerance: float = 1e-6,
    initialization_method: str = "closest_different",
    device: Optional[str] = None,
    verbose: bool = False
) -> Tuple[np.ndarray, np.ndarray]:

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device).eval()

    # --------------------------------------------------
    # Input normalization
    # --------------------------------------------------
    x = torch.tensor(samples, dtype=torch.float32, device=device)
    if x.ndim == 2:
        x = x.unsqueeze(1)  # (B, 1, L)

    B, C, L = x.shape
    x_orig = x.clone()
    print(x.shape)

    # --------------------------------------------------
    # Original predictions
    # --------------------------------------------------
    with torch.no_grad():
        logits_orig = model(x)
        probs_orig = torch.softmax(logits_orig, dim=1)
        labels_orig = probs_orig.argmax(dim=1)

    # --------------------------------------------------
    # Determine targets
    # --------------------------------------------------
    if targets is None:
        targets = torch.argsort(probs_orig, dim=1)[:, -2]
    else:
        targets = torch.tensor(targets, dtype=torch.long, device=device)

    active = (labels_orig != targets).float()  # (B,)

    # --------------------------------------------------
    # Dataset predictions (once)
    # --------------------------------------------------
    data = np.stack([np.asarray(d[0]) for d in dataset])[:,np.newaxis,:]
    data_t = torch.tensor(data, dtype=torch.float32, device=device)
    print(data_t.shape)

    with torch.no_grad():
        preds_data = torch.softmax(model(data_t), dim=1)
        labels_data = preds_data.argmax(dim=1).cpu().numpy()

    # --------------------------------------------------
    # Initialization
    # --------------------------------------------------
    x_cf = x_orig.clone()

    if initialization_method == "closest_different":
        for b in range(B):
            mask = labels_data == targets[b].item()
            if np.any(mask):
                candidates = data[mask]
                #print(type(candidates),type(samples))
                dists = np.sum(
                    (candidates.reshape(len(candidates), -1)
                     - samples[b].detach().cpu().numpy().reshape(1, -1)) ** 2,
                    axis=1,
                )
                x_cf[b] = torch.tensor(candidates[np.argmin(dists)], device=device)

    elif initialization_method == "random":
        idxs = np.random.randint(0, len(dataset), size=B)
        for b, i in enumerate(idxs):
            x_cf[b] = torch.tensor(dataset[i][0], device=device)

    # else: original → already set

    x_cf = x_cf.detach().requires_grad_(True)
    optimizer = optim.Adam([x_cf], lr=learning_rate)

    ce = nn.CrossEntropyLoss(reduction="none")

    best_cf = x_cf.detach().clone()
    best_validity = torch.zeros(B, device=device)

    prev_loss = torch.full((B,), float("inf"), device=device)

    # --------------------------------------------------
    # Optimization loop
    # --------------------------------------------------
    for it in range(max_iterations):
        optimizer.zero_grad()

        logits = model(x_cf)
        probs = torch.softmax(logits, dim=1)

        # ----------------------------------
        # Losses (per sample)
        # ----------------------------------
        cls_loss = ce(logits, targets)

        diff = (x_cf - x_orig).reshape(B, -1)
        proximity_loss = torch.mean(diff ** 2, dim=1)
        sparsity_loss = torch.mean(torch.abs(diff), dim=1)
        diversity_loss = 1.0 / (1.0 + proximity_loss)

        loss_per_sample = (
            cls_loss
            + lambda_proximity * proximity_loss
            + lambda_sparse * sparsity_loss
            + lambda_diversity * diversity_loss
        )

        loss = (loss_per_sample * active).sum() / (active.sum() + 1e-8)
        loss.backward()

        # Freeze inactive samples
        x_cf.grad *= active.view(B, 1, 1)
        optimizer.step()

        # ----------------------------------
        # Update best & stopping
        # ----------------------------------
        with torch.no_grad():
            validity = probs[torch.arange(B), targets]
            preds = probs.argmax(dim=1)

            improved = validity > best_validity
            best_validity[improved] = validity[improved]
            best_cf[improved] = x_cf.detach()[improved]

            converged = torch.abs(prev_loss - loss_per_sample) < tolerance
            solved = (preds == targets) & (it > 100)
            active[converged | solved] = 0.0

            prev_loss = loss_per_sample.detach()

        if verbose and it % 100 == 0:
            done = int((active == 0).sum().item())
            print(f"iter {it}: solved {done}/{B}")

        if active.sum() == 0:
            break

    # --------------------------------------------------
    # Final prediction
    # --------------------------------------------------
    with torch.no_grad():
        final_logits = model(best_cf)
        final_probs = torch.softmax(final_logits, dim=1)

    return (
        best_cf.detach(),      # (B, C, L)
        final_probs.detach(),  # (B, num_classes)
    )
