import torch
import torch.nn as nn
from tqdm import tqdm

class BasePerturbation(nn.Module):
    """
    Base class for batched perturbations.

    All subclasses MUST:
      - accept batched inputs (B, C, T)
      - return batched outputs (B, C, T)
      - return per-sample regularization of shape (B,)
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, T)
        return: (B, C, T)
        """
        raise NotImplementedError

    def regularization(self) -> torch.Tensor:
        """
        Returns:
            reg: (B,) tensor
        """
        # Safe default: zero regularization per sample
        return torch.zeros(
            getattr(self, "B", 1),
            device=next(self.parameters()).device
            if len(list(self.parameters())) > 0
            else "cpu",
        )
    

class STFTPerturbation(BasePerturbation):
    def __init__(
        self,
        B,
        C,
        T,
        n_fft=16,
        hop_length=None,
        win_length=None,
        init_scale=1e-2,
        window_fn=torch.hann_window,
    ):
        super().__init__()

        self.B = B
        self.C = C
        self.T = T
        self.n_fft = n_fft
        self.hop_length = hop_length or n_fft // 4
        self.win_length = win_length or n_fft
        self.window = window_fn(self.win_length)

        # Compute STFT shape
        dummy = torch.zeros(1, T)
        Z = torch.stft(
            dummy,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=True,
        )
        F, TT = Z.shape[-2], Z.shape[-1]

        # Per-sample learnable perturbation
        self.delta_real = nn.Parameter(init_scale * torch.randn(B, C, F, TT))
        self.delta_imag = nn.Parameter(init_scale * torch.randn(B, C, F, TT))
        self.mask_logits = nn.Parameter(torch.randn(B, C, F, TT))

        self.N = C * F * TT

    def forward(self, x):
        """
        x: (B, C, T)
        """
        B, C, T = x.shape
        window = self.window.to(x.device)

        # STFT
        Z = torch.stft(
            x.view(B * C, T),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            return_complex=True,
        )
        Z = Z.view(B, C, *Z.shape[-2:])

        mask = torch.sigmoid(self.mask_logits)
        delta = (self.delta_real + 1j * self.delta_imag) * mask
        Z_cf = Z + delta

        # inverse STFT
        x_cf = torch.istft(
            Z_cf.view(B * C, *Z_cf.shape[-2:]),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            length=T,
        )

        return x_cf.view(B, C, T)

    def regularization(self):
        """
        Returns: (B,)
        """
        mask = torch.sigmoid(self.mask_logits)

        sparsity = mask.sum(dim=(1, 2, 3)) / self.N
        energy = ((self.delta_real * mask) ** 2).sum(dim=(1, 2, 3))

        return sparsity + energy
    



    from tqdm import tqdm

class TSCounterfactualGenerator:
    def __init__(
        self,
        learner,
        perturbation_cls=STFTPerturbation,
        lam_perturb=1e-4,
        lam_time=0.0,
        lam_clf=1.0,
        lam_entropy=20,
        lr=1e-2,
        steps=500,
        device=None,
        **perturb_kwargs
    ):
        self.learner = learner
        self.model = learner.model.eval()
        self.device = device or learner.dls.device
        self.model.to(self.device)

        for p in self.model.parameters():
            p.requires_grad = False

        self.loss_func = learner.loss_func
        self.lam_perturb = lam_perturb
        self.lam_time = lam_time
        self.lam_clf = lam_clf
        self.lam_entropy = lam_entropy
        self.lr = lr
        self.steps = steps

        self.perturbation_cls = perturbation_cls
        self.perturb_kwargs = perturb_kwargs

    def generate(self, x, target):
        """
        x: (B, C, T)
        target: (B,)
        """
        x = x.to(self.device).float()
        target = target.to(self.device)

        B, C, T = x.shape

        perturb = self.perturbation_cls(
            B=B, C=C, T=T, **self.perturb_kwargs
        ).to(self.device)

        opt = torch.optim.Adam(perturb.parameters(), lr=self.lr)

        losses = []

        #Original
        logits_orig = self.model(x)
        y_orig = logits_orig.argmax(dim=1)   # (B,)


        for _ in tqdm(range(self.steps)):
            opt.zero_grad()

            # Counterfactual
            x_cf = perturb(x)
            logits = self.model(x_cf)


            #Classification loss (per-sample)
            clf_loss = self.loss_func(logits, target)

            # Distance regularization
            time_reg = (x_cf - x).abs().sum(dim=(1, 2))

            # Perturbation regularization
            pert_reg = perturb.regularization()

            # Entropy (encourage confident predictions)
            probas = torch.softmax(logits, dim=-1)
            entropy = (-(probas * torch.log(probas + 1e-8)).sum(dim=1))#.pow(0.5)

            # p_orig = probas.gather(1, y_orig[:, None]).squeeze(1)   # (B,)
            # p_tgt  = probas.gather(1, target[:, None]).squeeze(1)   # (B,)
            # #Normalize to probability dist
            # p_pair = torch.stack([p_orig, p_tgt], dim=1)             # (B, 2)
            # p_pair = p_pair / (p_pair.sum(dim=1, keepdim=True) + 1e-8)
            # entropy = -(p_pair * torch.log(p_pair + 1e-8)).sum(dim=1)  # (B,)

            # Adaptive lambdas (optional)
            lr_clf = self.lam_clf if self.lam_clf is not None else 1.0
            lr_time = self.lam_time if self.lam_time is not None else 0.05
            lr_perturb = self.lam_perturb if self.lam_perturb is not None else 0.5
            lr_entropy = self.lam_entropy if self.lam_entropy is not None else 20

            # Total loss (mean over batch)
            loss = (
                lr_clf * clf_loss.pow(2)
                + lr_perturb * pert_reg.pow(2)
                + lr_time * time_reg.pow(2)
                - lr_entropy * entropy
            ).sum()

            loss.backward()
            opt.step()

            losses.append(loss.item())

            # Small-change snapping (optional)
            x_cf = torch.where(
                (x_cf - x).pow(2) < 1e-4, x, x_cf
            )

        return {
            "x_cf": x_cf.detach(),
            "losses": losses,
            "perturbation": perturb,
        }


