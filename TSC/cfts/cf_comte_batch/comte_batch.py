import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Optional, Tuple


def cf_ts(
    samples: np.ndarray,                 # (B, C, L)
    model: nn.Module,
    targets: Optional[np.ndarray] = None, # (B,)
    lambda_reg: float = 0.01,
    lambda_sparse: float = 0.001,
    lambda_smooth: float = 0.01,
    lambda_temporal: float = 0.005,
    learning_rate: float = 0.05,
    max_iterations: int = 3000,
    tolerance: float = 1e-4,
    device: Optional[str] = None,
    verbose: bool = False,
    dataset = None
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

    # --------------------------------------------------
    # Original predictions
    # --------------------------------------------------
    with torch.no_grad():
        logits_orig = model(x)
        probs_orig = torch.softmax(logits_orig, dim=1)
        labels_orig = probs_orig.argmax(dim=1)

    if targets is None:
        targets = torch.argsort(probs_orig, dim=1)[:, -2]
    else:
        targets = torch.tensor(targets, device=device, dtype=torch.long)

    # Mask samples that already match target
    active = (labels_orig != targets).float()  # (B,)
    #print(active)
    # --------------------------------------------------
    # Temporal reference statistics
    # --------------------------------------------------
    with torch.no_grad():
        orig_diff1 = x_orig[:, :, 1:] - x_orig[:, :, :-1]
        orig_diff2 = orig_diff1[:, :, 1:] - orig_diff1[:, :, :-1]

    # --------------------------------------------------
    # Initialize CFs
    # --------------------------------------------------
    x_cf = x_orig.clone().detach().requires_grad_(True)
    optimizer = optim.Adam([x_cf], lr=learning_rate)

    best_cf = x_cf.clone().detach()
    best_validity = torch.zeros(B, device=device)

    # --------------------------------------------------
    # Training phases
    # --------------------------------------------------
    p1 = max_iterations // 3
    p2 = 2 * max_iterations // 3

    for it in range(max_iterations):
        optimizer.zero_grad()

        # -------------------------
        # Phase scheduling
        # -------------------------
        if it < p1:
            l_reg = l_sparse = l_smooth = l_temp = 0.0
        elif it < p2:
            l_reg = 0.5 * lambda_reg
            l_sparse = 0.5 * lambda_sparse
            l_smooth = lambda_smooth
            l_temp = 0.5 * lambda_temporal
        else:
            l_reg = lambda_reg
            l_sparse = lambda_sparse
            l_smooth = lambda_smooth
            l_temp = lambda_temporal

        # -------------------------
        # Forward
        # -------------------------
        logits = model(x_cf)
        log_probs = torch.log_softmax(logits, dim=1)

        # Prediction loss (per sample)
        pred_loss = -log_probs[torch.arange(B), targets]

        # Proximity (L2)
        dist_l2 = torch.norm((x_cf - x_orig).reshape(B, -1), p=2, dim=1)

        # Sparsity (L1)
        dist_l1 = torch.norm((x_cf - x_orig).reshape(B, -1), p=1, dim=1)

        # Temporal smoothness
        cf_diff1 = x_cf[:, :, 1:] - x_cf[:, :, :-1]
        smooth_loss = torch.norm(cf_diff1.reshape(B, -1), p=2, dim=1)

        # Temporal consistency
        cf_diff2 = cf_diff1[:, :, 1:] - cf_diff1[:, :, :-1]
        temp_loss = torch.norm(
            (cf_diff2 - orig_diff2).reshape(B, -1), p=2, dim=1
        )

        # -------------------------
        # Total loss (per sample)
        # -------------------------
        loss_per_sample = (
            pred_loss
            + l_reg * dist_l2
            + l_sparse * dist_l1
            + l_smooth * smooth_loss
            + l_temp * temp_loss
        )

        # Mask inactive samples
        loss = (loss_per_sample * active).sum() / (active.sum() + 1e-8)
        loss.backward()

        # Zero gradients for inactive samples
        x_cf.grad *= active.view(B, 1, 1)

        optimizer.step()

        # -------------------------
        # Validity check
        # -------------------------
        with torch.no_grad():
            probs = torch.softmax(logits, dim=1)
            validity = probs[torch.arange(B), targets]
            preds = probs.argmax(dim=1)

            improved = validity > best_validity
            best_validity[improved] = validity[improved]
            best_cf[improved] = x_cf.detach()[improved]

            # Freeze solved samples
            solved = (preds == targets) & (validity > 0.95)
            active[solved] = 0.0

        if verbose and it % 500 == 0:
            done = int((active == 0).sum().item())
            print(f"iter {it}: solved {done}/{B}")

        if active.sum() == 0:
            if verbose:
                print(f"All CFs found at iteration {it}")
            break

    # --------------------------------------------------
    # Final prediction
    # --------------------------------------------------
    with torch.no_grad():
        final_logits = model(best_cf)
        final_probs = torch.softmax(final_logits, dim=1)

    return (
        best_cf.detach(),      # (B, C, L)
        final_probs.detach()   # (B, num_classes)
    )
