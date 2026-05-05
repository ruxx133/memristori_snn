
# Full script: strong continuous conv-SNN backbone + hybrid DSPR head comparison
import argparse
import math
import random
import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import functional as TF


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(req: str) -> str:
    if req != "auto":
        return req
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_sheet_name(sheet_arg):
    if sheet_arg is None:
        return 0
    try:
        return int(sheet_arg)
    except (TypeError, ValueError):
        return sheet_arg


@dataclass
class Segment:
    sid: int
    start_idx: int
    end_idx: int
    voltage: float
    t0: float
    t1: float
    duration: float
    current_median: float
    current_last: float
    n: int
    note: str
    kind: str


@dataclass
class DSPRParams:
    G_low: float
    G_high: float
    kappa_plus: float
    kappa_zero: float
    sigma_plus: float
    sigma_zero: float
    V_read: float
    V_set: float
    V_reset: float


def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def load_protocol_xlsx(xlsx_path: str, sheet_name=0):
    try:
        read_sheet = 0 if sheet_name is None else sheet_name
        df = pd.read_excel(xlsx_path, sheet_name=read_sheet, engine="openpyxl")
    except ImportError as e:
        raise ImportError(
            "Missing dependency 'openpyxl'. Install it with:\n"
            "  conda install openpyxl\n"
            "or\n"
            "  python -m pip install openpyxl"
        ) from e

    if isinstance(df, dict):
        if not df:
            raise ValueError(f"No sheets found in Excel file: {xlsx_path}")
        first_sheet_name = next(iter(df.keys()))
        print(f"[INFO] Multiple sheets found. Using first sheet: {first_sheet_name}")
        df = df[first_sheet_name]

    df.columns = [str(c).strip() for c in df.columns]
    required = {"Time,s", "VOLTAGE", "CURRENT"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns {missing}. Found columns: {list(df.columns)}")
    if "NOTES" not in df.columns:
        df["NOTES"] = ""
    return df


def classify_segment(voltage: float, note: str, v_read_nominal: float = 1.0) -> str:
    txt = normalize_text(note)
    reset_keywords = ["reset", "reseteaza", "reseta", "reseta m", "puls de reset", "ca sa resetam"]
    read_keywords = ["read", "citire"]
    set_keywords = ["set", "write voltage la 5", "write voltage set", "write voltage la 5v"]

    is_read = (abs(voltage - v_read_nominal) < 0.25) or any(k in txt for k in read_keywords)
    if is_read:
        return "read"

    is_reset = (abs(voltage) < 0.15) and any(k in txt for k in reset_keywords)
    if is_reset:
        return "reset"

    is_set = (voltage >= 4.5) or any(k in txt for k in set_keywords)
    if is_set:
        return "set"

    return "other"


def segment_protocol(df: pd.DataFrame) -> List[Segment]:
    t = df["Time,s"].astype(float).to_numpy()
    v = df["VOLTAGE"].astype(float).to_numpy()
    i = df["CURRENT"].astype(float).to_numpy()
    notes = df["NOTES"].fillna("").astype(str).tolist()

    seg_id = np.zeros(len(df), dtype=np.int32)
    seg_id[1:] = np.cumsum(np.abs(v[1:] - v[:-1]) > 1e-12)

    segments: List[Segment] = []
    for sid in np.unique(seg_id):
        idx = np.where(seg_id == sid)[0]
        note = " ".join(n for n in (notes[j] for j in idx) if str(n).strip())
        voltage = float(np.median(v[idx]))
        current_median = float(np.median(i[idx]))
        current_last = float(i[idx[-1]])
        duration = float(t[idx[-1]] - t[idx[0]])
        kind = classify_segment(voltage, note)
        segments.append(
            Segment(
                sid=int(sid),
                start_idx=int(idx[0]),
                end_idx=int(idx[-1]),
                voltage=voltage,
                t0=float(t[idx[0]]),
                t1=float(t[idx[-1]]),
                duration=duration,
                current_median=current_median,
                current_last=current_last,
                n=len(idx),
                note=note,
                kind=kind,
            )
        )
    return segments


def read_conductance_from_segment(seg: Segment, v_read_nominal: float = 1.0) -> float:
    if abs(seg.voltage) < 1e-12:
        raise ValueError("Cannot compute conductance from zero-voltage segment.")
    return seg.current_median / float(seg.voltage if abs(seg.voltage) > 1e-12 else v_read_nominal)


def robust_plateaus(read_G: np.ndarray) -> Tuple[float, float]:
    G_low = float(np.min(read_G))
    G_high = float(np.max(read_G))
    if G_high <= G_low:
        raise ValueError("Degenerate read conductance range.")
    return G_low, G_high


def normalize_G(G: float, G_low: float, G_high: float) -> float:
    return float(np.clip((G - G_low) / (G_high - G_low + 1e-12), 0.0, 1.0))


def estimate_kappas(segments: List[Segment], G_low: float, G_high: float) -> Tuple[float, float, float, float]:
    set_rates = []
    reset_rates = []
    set_res = []
    reset_res = []

    for a, b, c in zip(segments[:-2], segments[1:-1], segments[2:]):
        if a.kind != "read" or c.kind != "read":
            continue
        G0 = read_conductance_from_segment(a)
        G1 = read_conductance_from_segment(c)
        z0 = normalize_G(G0, G_low, G_high)
        z1 = normalize_G(G1, G_low, G_high)
        dt = max(float(b.duration), 1e-6)

        if b.kind == "set" and z1 > z0 and z0 < 0.999 and z1 < 0.999999:
            num = max(1.0 - z1, 1e-8)
            den = max(1.0 - z0, 1e-8)
            if num < den:
                k = -math.log(num / den) / dt
                if np.isfinite(k) and k > 0:
                    set_rates.append(k)
        elif b.kind == "reset" and z1 < z0 and z0 > 1e-6 and z1 > 1e-6:
            k = -math.log(max(z1, 1e-8) / max(z0, 1e-8)) / dt
            if np.isfinite(k) and k > 0:
                reset_rates.append(k)

    if len(set_rates) == 0:
        raise ValueError("Could not estimate kappa_plus from read->set->read triples.")
    if len(reset_rates) == 0:
        raise ValueError("Could not estimate kappa_zero from read->reset->read triples.")

    kappa_plus = float(np.median(set_rates))
    kappa_zero = float(np.median(reset_rates))

    for a, b, c in zip(segments[:-2], segments[1:-1], segments[2:]):
        if a.kind != "read" or c.kind != "read":
            continue
        G0 = read_conductance_from_segment(a)
        G1 = read_conductance_from_segment(c)
        z0 = normalize_G(G0, G_low, G_high)
        z1 = normalize_G(G1, G_low, G_high)
        dt = max(float(b.duration), 1e-6)
        if b.kind == "set":
            zhat = 1.0 - (1.0 - z0) * math.exp(-kappa_plus * dt)
            set_res.append(z1 - zhat)
        elif b.kind == "reset":
            zhat = z0 * math.exp(-kappa_zero * dt)
            reset_res.append(z1 - zhat)

    sigma_plus = float(np.std(set_res)) if len(set_res) > 1 else 0.0
    sigma_zero = float(np.std(reset_res)) if len(reset_res) > 1 else 0.0
    return kappa_plus, kappa_zero, sigma_plus, sigma_zero


def fit_dspr_from_xlsx(xlsx_path: str, sheet_name: Optional[str] = None) -> Tuple[DSPRParams, List[Segment]]:
    df = load_protocol_xlsx(xlsx_path, sheet_name=sheet_name)
    segments = segment_protocol(df)

    read_segments = [s for s in segments if s.kind == "read" and abs(s.voltage) > 1e-12]
    if len(read_segments) < 4:
        raise ValueError("Need several read segments to estimate G_low/G_high.")

    read_G = np.array([read_conductance_from_segment(s) for s in read_segments], dtype=np.float64)
    G_low, G_high = robust_plateaus(read_G)
    kappa_plus, kappa_zero, sigma_plus, sigma_zero = estimate_kappas(segments, G_low, G_high)

    V_read_values = [s.voltage for s in read_segments]
    V_set_values = [s.voltage for s in segments if s.kind == "set"]
    V_reset_values = [s.voltage for s in segments if s.kind == "reset"]

    params = DSPRParams(
        G_low=G_low,
        G_high=G_high,
        kappa_plus=kappa_plus,
        kappa_zero=kappa_zero,
        sigma_plus=sigma_plus,
        sigma_zero=sigma_zero,
        V_read=float(np.median(V_read_values)) if V_read_values else 1.0,
        V_set=float(np.median(V_set_values)) if V_set_values else 5.0,
        V_reset=float(np.median(V_reset_values)) if V_reset_values else 0.0,
    )
    return params, segments


def emnist_fix(x: torch.Tensor) -> torch.Tensor:
    x = torch.rot90(x, k=-1, dims=[1, 2])
    x = torch.flip(x, dims=[2])
    return x


def build_transform(dataset_name: str):
    name_l = dataset_name.lower()
    if "emnist" in name_l:
        return transforms.Compose([transforms.ToTensor(), transforms.Lambda(emnist_fix)])
    return transforms.Compose([transforms.ToTensor()])


def get_dataset(name: str, data_dir: str, train: bool):
    tfm = build_transform(name)
    name_l = name.lower()
    try:
        if name_l == "fashionmnist":
            return datasets.FashionMNIST(root=data_dir, train=train, download=True, transform=tfm), 10
        if name_l == "kmnist":
            return datasets.KMNIST(root=data_dir, train=train, download=True, transform=tfm), 10
        if name_l == "mnist":
            return datasets.MNIST(root=data_dir, train=train, download=True, transform=tfm), 10
        if name_l == "emnist-balanced":
            return datasets.EMNIST(root=data_dir, split="balanced", train=train, download=True, transform=tfm), 47
        if name_l == "emnist-letters":
            base = datasets.EMNIST(root=data_dir, split="letters", train=train, download=True, transform=tfm)
            class ShiftLetters(torch.utils.data.Dataset):
                def __init__(self, ds): self.ds = ds
                def __len__(self): return len(self.ds)
                def __getitem__(self, idx):
                    x, y = self.ds[idx]
                    return x, int(y) - 1
            return ShiftLetters(base), 26
        if name_l == "qmnist":
            what = "train" if train else "test10k"
            return datasets.QMNIST(root=data_dir, what=what, download=True, transform=tfm, compat=True), 10
    except Exception as e:
        raise RuntimeError(f"Dataset {name} could not be loaded/downloaded: {e}")
    raise ValueError(f"Unsupported dataset: {name}")


def safe_collate(batch):
    xs, ys = zip(*batch)
    xs_out = []
    for x in xs:
        xs_out.append(x if isinstance(x, torch.Tensor) else TF.to_tensor(x))
    ys_out = []
    for y in ys:
        ys_out.append(int(y.item()) if isinstance(y, torch.Tensor) else int(y))
    return torch.stack(xs_out, dim=0), torch.tensor(ys_out, dtype=torch.long)


def subset_dataset(ds, subset: int):
    if subset is None or subset <= 0 or subset >= len(ds):
        return ds
    return torch.utils.data.Subset(ds, range(subset))


def make_loader(ds, batch_size: int, train: bool):
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=train,
        drop_last=train,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        collate_fn=safe_collate,
    )


class FastSigmoidSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, slope: float):
        ctx.save_for_backward(x)
        ctx.slope = float(slope)
        return (x > 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (x,) = ctx.saved_tensors
        slope = ctx.slope
        grad = slope / (1.0 + torch.abs(slope * x)) ** 2
        return grad_output * grad, None


def spike_fn(x: torch.Tensor, slope: float) -> torch.Tensor:
    return FastSigmoidSpike.apply(x, slope)


class LearnableLIFCell(nn.Module):
    def __init__(self, beta_init=0.90, threshold=0.70, reset="subtract", spike_slope=12.0):
        super().__init__()
        beta_init = float(np.clip(beta_init, 1e-4, 1.0 - 1e-4))
        rho = math.log(beta_init / (1.0 - beta_init))
        self.rho = nn.Parameter(torch.tensor(rho, dtype=torch.float32))
        self.threshold = float(threshold)
        self.reset = reset
        self.spike_slope = float(spike_slope)

    def beta(self):
        return torch.sigmoid(self.rho)

    def forward(self, input_current: torch.Tensor, mem: torch.Tensor):
        mem = self.beta() * mem + input_current
        spk = spike_fn(mem - self.threshold, self.spike_slope)
        if self.reset == "subtract":
            mem = mem - spk * self.threshold
        elif self.reset == "zero":
            mem = mem * (1.0 - spk)
        return spk, mem


class LearnableLICell(nn.Module):
    def __init__(self, beta_init=0.85):
        super().__init__()
        beta_init = float(np.clip(beta_init, 1e-4, 1.0 - 1e-4))
        rho = math.log(beta_init / (1.0 - beta_init))
        self.rho = nn.Parameter(torch.tensor(rho, dtype=torch.float32))

    def beta(self):
        return torch.sigmoid(self.rho)

    def forward(self, input_current: torch.Tensor, mem: torch.Tensor):
        return self.beta() * mem + input_current


class ConvSNNBackbone(nn.Module):
    def __init__(
        self,
        feature_dim: int = 256,
        beta_h: float = 0.90,
        thr_h: float = 0.70,
        beta_feat: float = 0.90,
        spike_slope: float = 12.0,
        syn_scale_conv1: float = 0.60,
        syn_scale_conv2: float = 0.45,
        syn_scale_fc1: float = 0.28,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        self.fc1 = nn.Linear(64 * 7 * 7, feature_dim, bias=False)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm1d(feature_dim)

        self.lif1 = LearnableLIFCell(beta_init=beta_h, threshold=thr_h, spike_slope=spike_slope)
        self.lif2 = LearnableLIFCell(beta_init=beta_h, threshold=thr_h, spike_slope=spike_slope)
        self.lif3 = LearnableLIFCell(beta_init=beta_feat, threshold=thr_h, spike_slope=spike_slope)

        self.pool = nn.MaxPool2d(2)
        self.syn_scale_conv1 = float(syn_scale_conv1)
        self.syn_scale_conv2 = float(syn_scale_conv2)
        self.syn_scale_fc1 = float(syn_scale_fc1)
        self.feature_dim = int(feature_dim)

    def forward(self, spk_in: torch.Tensor):
        T, B = spk_in.shape[0], spk_in.shape[1]
        mem1 = torch.zeros(B, 32, 28, 28, device=spk_in.device)
        mem2 = torch.zeros(B, 64, 14, 14, device=spk_in.device)
        mem3 = torch.zeros(B, self.feature_dim, device=spk_in.device)

        feat_rec = []
        fr_accum = 0.0

        for t in range(T):
            x = spk_in[t]
            cur1 = self.bn1(self.syn_scale_conv1 * self.conv1(x))
            spk1, mem1 = self.lif1(cur1, mem1)
            x1 = self.pool(spk1)

            cur2 = self.bn2(self.syn_scale_conv2 * self.conv2(x1))
            spk2, mem2 = self.lif2(cur2, mem2)
            x2 = self.pool(spk2)

            x2 = x2.flatten(1)
            cur3 = self.bn3(self.syn_scale_fc1 * self.fc1(x2))
            spk3, mem3 = self.lif3(cur3, mem3)

            feat_rec.append(spk3)
            fr_accum = fr_accum + spk1.mean() + spk2.mean() + spk3.mean()

        feat_rec = torch.stack(feat_rec)
        mean_fr = fr_accum / (3.0 * T)
        return feat_rec, mean_fr


class ContinuousHead(nn.Module):
    def __init__(self, feature_dim: int, n_out: int):
        super().__init__()
        self.fc = nn.Linear(feature_dim, n_out, bias=False)
        self.li_out = LearnableLICell(beta_init=0.85)

    def forward(self, feat_rec: torch.Tensor):
        T, B, _ = feat_rec.shape
        mem_out = torch.zeros(B, self.fc.out_features, device=feat_rec.device)
        mem_rec = []
        for t in range(T):
            cur = self.fc(feat_rec[t])
            mem_out = self.li_out(cur, mem_out)
            mem_rec.append(mem_out)
        return torch.stack(mem_rec)


class DSPRHead(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        n_out: int,
        dspr: DSPRParams,
        wmax: float = 1.0,
        init_mode: str = "mid",
        init_spread: float = 0.03,
        history_lambda: float = 0.0,
        tau_h: float = 1.0,
        history_increment: float = 1.0,
        noise_on: bool = False,
        leak_dt: float = 0.0,
        delta_z_cap: float = 0.03,
        li_beta_init: float = 0.85,
    ):
        super().__init__()
        self.wmax = float(wmax)
        self.kappa_plus = float(dspr.kappa_plus)
        self.kappa_zero = float(dspr.kappa_zero)
        self.sigma_plus = float(dspr.sigma_plus)
        self.sigma_zero = float(dspr.sigma_zero)
        self.history_lambda = float(history_lambda)
        self.tau_h = float(tau_h)
        self.history_increment = float(history_increment)
        self.noise_on = bool(noise_on)
        self.leak_dt = float(leak_dt)
        self.delta_z_cap = float(delta_z_cap)
        self.eps = 1e-8

        if init_mode == "low":
            mean_pos, mean_neg = 0.20, 0.10
        elif init_mode == "high":
            mean_pos, mean_neg = 0.75, 0.25
        else:
            mean_pos, mean_neg = 0.55, 0.45

        zpos = torch.empty(n_out, feature_dim).normal_(mean=mean_pos, std=init_spread).clamp_(0.0, 1.0)
        zneg = torch.empty(n_out, feature_dim).normal_(mean=mean_neg, std=init_spread).clamp_(0.0, 1.0)

        self.z_pos = nn.Parameter(zpos)
        self.z_neg = nn.Parameter(zneg)
        self.register_buffer("h_pos", torch.zeros(n_out, feature_dim))
        self.register_buffer("h_neg", torch.zeros(n_out, feature_dim))

        self.li_out = LearnableLICell(beta_init=li_beta_init)

    def effective_weight(self) -> torch.Tensor:
        return self.wmax * (self.z_pos - self.z_neg)

    def forward(self, feat_rec: torch.Tensor):
        T, B, _ = feat_rec.shape
        mem_out = torch.zeros(B, self.z_pos.shape[0], device=feat_rec.device)
        mem_rec = []
        W = self.effective_weight()
        for t in range(T):
            cur = feat_rec[t] @ W.t()
            mem_out = self.li_out(cur, mem_out)
            mem_rec.append(mem_out)
        return torch.stack(mem_rec)

    def _kappa_eff(self, h: torch.Tensor, base: float) -> torch.Tensor:
        if self.history_lambda == 0.0:
            return torch.full_like(h, fill_value=base)
        return base * (1.0 + self.history_lambda * h)

    def _apply_set(self, z: torch.Tensor, h: torch.Tensor, dz_target: torch.Tensor):
        if dz_target.max().item() <= 0.0:
            return z, h
        kappa_eff = self._kappa_eff(h, self.kappa_plus)
        room = (1.0 - z).clamp(min=self.eps)
        dz_target = torch.minimum(dz_target, room * (1.0 - 1e-6))
        tau = -torch.log1p(-(dz_target / room).clamp(max=1.0 - 1e-6)) / kappa_eff.clamp(min=self.eps)
        z_new = 1.0 - (1.0 - z) * torch.exp(-kappa_eff * tau)
        if self.noise_on and self.sigma_plus > 0.0:
            z_new = z_new + self.sigma_plus * torch.sqrt((1.0 - z).clamp(min=0.0)) * torch.randn_like(z)
        z_new = z_new.clamp(0.0, 1.0)
        if self.history_lambda != 0.0:
            if self.tau_h > 0:
                h_new = torch.exp(-tau / self.tau_h) * h + self.history_increment
            else:
                h_new = h + self.history_increment
        else:
            h_new = h
        return z_new, h_new

    def _apply_leak(self, z: torch.Tensor, h: torch.Tensor):
        if self.leak_dt <= 0.0:
            return z, h
        kappa_eff = self._kappa_eff(h, self.kappa_zero)
        z_new = z * torch.exp(-kappa_eff * self.leak_dt)
        if self.noise_on and self.sigma_zero > 0.0:
            z_new = z_new + self.sigma_zero * torch.sqrt(z.clamp(min=0.0)) * torch.randn_like(z)
        z_new = z_new.clamp(0.0, 1.0)
        if self.history_lambda != 0.0 and self.tau_h > 0:
            h_new = torch.exp(torch.full_like(h, -self.leak_dt / self.tau_h)) * h
        else:
            h_new = h
        return z_new, h_new

    @torch.no_grad()
    def dspr_step(self, eta_net: float) -> None:
        if self.z_pos.grad is None or self.z_neg.grad is None:
            return
        grad_w = 0.5 * (self.z_pos.grad / self.wmax - self.z_neg.grad / self.wmax)
        desired_dw = -float(eta_net) * grad_w
        desired_dz = (desired_dw.abs() / max(self.wmax, self.eps)).clamp(min=0.0, max=self.delta_z_cap)

        pos_mask = desired_dw > 0.0
        neg_mask = desired_dw < 0.0

        if pos_mask.any():
            dz_pos = torch.where(pos_mask, desired_dz, torch.zeros_like(desired_dz))
            z_new, h_new = self._apply_set(self.z_pos, self.h_pos, dz_pos)
            self.z_pos.copy_(torch.where(pos_mask, z_new, self.z_pos))
            self.h_pos.copy_(torch.where(pos_mask, h_new, self.h_pos))

        if neg_mask.any():
            dz_neg = torch.where(neg_mask, desired_dz, torch.zeros_like(desired_dz))
            z_new, h_new = self._apply_set(self.z_neg, self.h_neg, dz_neg)
            self.z_neg.copy_(torch.where(neg_mask, z_new, self.z_neg))
            self.h_neg.copy_(torch.where(neg_mask, h_new, self.h_neg))

        zpl, hpl = self._apply_leak(self.z_pos, self.h_pos)
        znl, hnl = self._apply_leak(self.z_neg, self.h_neg)
        self.z_pos.copy_(zpl.clamp(0.0, 1.0))
        self.z_neg.copy_(znl.clamp(0.0, 1.0))
        self.h_pos.copy_(hpl)
        self.h_neg.copy_(hnl)

        if self.z_pos.grad is not None:
            self.z_pos.grad.zero_()
        if self.z_neg.grad is not None:
            self.z_neg.grad.zero_()

    @torch.no_grad()
    def initialize_from_continuous_weight(self, W: torch.Tensor):
        Wc = torch.clamp(W / max(self.wmax, 1e-8), -1.0, 1.0)
        pos = torch.clamp(Wc, min=0.0)
        neg = torch.clamp(-Wc, min=0.0)
        z_base = 0.45
        self.z_pos.copy_(torch.clamp(z_base + pos, 0.0, 1.0))
        self.z_neg.copy_(torch.clamp(z_base + neg, 0.0, 1.0))
        self.h_pos.zero_()
        self.h_neg.zero_()


class ContinuousBackboneModel(nn.Module):
    def __init__(self, backbone: ConvSNNBackbone, head: ContinuousHead):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, spk_in: torch.Tensor):
        feat_rec, fr = self.backbone(spk_in)
        mem_rec = self.head(feat_rec)
        return mem_rec, fr


class HybridDSPRModel(nn.Module):
    def __init__(self, backbone: ConvSNNBackbone, head: DSPRHead):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, spk_in: torch.Tensor):
        feat_rec, fr = self.backbone(spk_in)
        mem_rec = self.head(feat_rec)
        return mem_rec, fr


def encode_rate(x: torch.Tensor, T: int, rate_scale: float) -> torch.Tensor:
    p = torch.clamp(x * rate_scale, 0.0, 1.0)
    return (torch.rand((T,) + x.shape, device=x.device) < p.unsqueeze(0)).float()


def encode_latency(x: torch.Tensor, T: int, thr: float) -> torch.Tensor:
    B, C, H, W = x.shape
    flat = x.view(B, -1)
    mask = flat >= thr
    t_spike = ((1.0 - flat) * (T - 1)).clamp(0, T - 1).long()
    spk = torch.zeros(B, flat.shape[1], T, device=x.device)
    spk.scatter_(2, t_spike.unsqueeze(-1), 1.0)
    spk = spk * mask.unsqueeze(-1).float()
    spk = spk.permute(2, 0, 1).contiguous()
    return spk.view(T, B, C, H, W)


def output_logits(mem_out_rec: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "mean":
        return mem_out_rec.mean(dim=0)
    if mode == "sum":
        return mem_out_rec.sum(dim=0) / mem_out_rec.shape[0]
    if mode == "max":
        return mem_out_rec.max(dim=0).values
    if mode == "last":
        return mem_out_rec[-1]
    raise ValueError("mode must be one of sum|mean|max|last")


def temporal_ce_loss(mem_out_rec: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    losses = [F.cross_entropy(mem_out_rec[t], targets) for t in range(mem_out_rec.shape[0])]
    return torch.stack(losses).mean()


def compute_losses(mem_rec: torch.Tensor, y: torch.Tensor, mem_logits_mode: str, temporal_loss_weight: float):
    logits = output_logits(mem_rec, mem_logits_mode)
    loss_readout = F.cross_entropy(logits, y)
    loss_temporal = temporal_ce_loss(mem_rec, y)
    loss = (1.0 - temporal_loss_weight) * loss_readout + temporal_loss_weight * loss_temporal
    return loss, logits


@torch.no_grad()
def eval_model(model, loader, device, args):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_n = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = torch.as_tensor(yb, device=device, dtype=torch.long)
        if args.encoding == "rate":
            spk_in = encode_rate(xb, args.T, args.rate_scale)
        else:
            spk_in = encode_latency(xb, args.T, args.latency_thr)
        mem_rec, _ = model(spk_in)
        loss, logits = compute_losses(mem_rec, yb, args.mem_logits, args.temporal_loss_weight)
        total_loss += float(loss.item()) * xb.size(0)
        total_correct += int((logits.argmax(dim=1) == yb).sum().item())
        total_n += xb.size(0)
    return {"loss": float(total_loss) / max(total_n, 1), "acc": float(total_correct) / max(total_n, 1)}


def pretrain_continuous(backbone_model: ContinuousBackboneModel, train_loader, test_loader, device, args):
    params = list(backbone_model.parameters())
    opt = torch.optim.Adam(params, lr=args.pretrain_lr, weight_decay=args.weight_decay)
    best_acc = -1.0
    best_state = None

    print("\n=== Phase 1: continuous pretraining ===")
    for epoch in range(1, args.pretrain_epochs + 1):
        backbone_model.train()
        total_loss = 0.0
        total_correct = 0
        total_n = 0
        total_fr = 0.0
        printed_debug = False

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = torch.as_tensor(yb, device=device, dtype=torch.long)
            if args.encoding == "rate":
                spk_in = encode_rate(xb, args.T, args.rate_scale)
            else:
                spk_in = encode_latency(xb, args.T, args.latency_thr)

            opt.zero_grad(set_to_none=True)
            mem_rec, fr = backbone_model(spk_in)
            loss, logits = compute_losses(mem_rec, yb, args.mem_logits, args.temporal_loss_weight)
            if args.fr_lambda > 0.0:
                loss = loss + args.fr_lambda * (fr - args.fr_target) ** 2
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            opt.step()

            if not printed_debug:
                printed_debug = True
                print(f"[PRETRAIN DEBUG] input fr: {float(spk_in.mean().item()):.4f}")
                print(f"[PRETRAIN DEBUG] hidden fr: {float(fr.item()):.4f}")
                print(f"[PRETRAIN DEBUG] logits |mean|: {float(logits.abs().mean().item()):.4f}")

            total_loss += float(loss.item()) * xb.size(0)
            total_correct += int((logits.argmax(dim=1) == yb).sum().item())
            total_n += xb.size(0)
            total_fr += float(fr.item()) * xb.size(0)

        tr_loss = float(total_loss) / max(total_n, 1)
        tr_acc = float(total_correct) / max(total_n, 1)
        tr_fr = float(total_fr) / max(total_n, 1)
        te = eval_model(backbone_model, test_loader, device, args)

        print(f"[PRETRAIN] Epoch {epoch:02d} | train loss {tr_loss:.4f} acc {tr_acc:.4f} fr_hid {tr_fr:.4f} | test loss {te['loss']:.4f} acc {te['acc']:.4f}")

        if te["acc"] > best_acc:
            best_acc = te["acc"]
            best_state = {
                "backbone": backbone_model.backbone.state_dict(),
                "head": backbone_model.head.state_dict(),
            }

    backbone_model.backbone.load_state_dict(best_state["backbone"])
    backbone_model.head.load_state_dict(best_state["head"])
    return best_acc, best_state


def finetune_dspr_head(hybrid_model: HybridDSPRModel, train_loader, test_loader, device, args):
    for p in hybrid_model.backbone.parameters():
        p.requires_grad = False
    for m in [hybrid_model.backbone.bn1, hybrid_model.backbone.bn2, hybrid_model.backbone.bn3,
              hybrid_model.backbone.lif1, hybrid_model.backbone.lif2, hybrid_model.backbone.lif3]:
        for p in m.parameters():
            p.requires_grad = True

    aux_params = []
    for name, p in hybrid_model.named_parameters():
        if name.startswith("head.z_pos") or name.startswith("head.z_neg"):
            continue
        if p.requires_grad:
            aux_params.append(p)

    aux_opt = torch.optim.Adam(aux_params, lr=args.finetune_aux_lr, weight_decay=args.weight_decay)
    best_acc = -1.0
    best_state = None

    print("\n=== Phase 3: DSPR-aware head fine-tuning ===")
    for epoch in range(1, args.finetune_epochs + 1):
        hybrid_model.train()
        total_loss = 0.0
        total_correct = 0
        total_n = 0
        total_fr = 0.0
        eta_eff = eta_schedule(epoch, args.eta_net, args.eta_warmup_epochs)
        printed_debug = False

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = torch.as_tensor(yb, device=device, dtype=torch.long)
            if args.encoding == "rate":
                spk_in = encode_rate(xb, args.T, args.rate_scale)
            else:
                spk_in = encode_latency(xb, args.T, args.latency_thr)

            hybrid_model.zero_grad(set_to_none=True)
            aux_opt.zero_grad(set_to_none=True)

            mem_rec, fr = hybrid_model(spk_in)
            loss, logits = compute_losses(mem_rec, yb, args.mem_logits, args.temporal_loss_weight)
            if args.fr_lambda > 0.0:
                loss = loss + args.fr_lambda * (fr - args.fr_target) ** 2
            loss.backward()

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(aux_params + [hybrid_model.head.z_pos, hybrid_model.head.z_neg], args.grad_clip)

            aux_opt.step()
            hybrid_model.head.dspr_step(eta_eff)

            if not printed_debug:
                printed_debug = True
                print(f"[DSPR DEBUG] input fr: {float(spk_in.mean().item()):.4f}")
                print(f"[DSPR DEBUG] hidden fr: {float(fr.item()):.4f}")
                print(f"[DSPR DEBUG] logits |mean|: {float(logits.abs().mean().item()):.4f}")
                print(f"[DSPR DEBUG] eta_eff: {eta_eff:.4e}")

            total_loss += float(loss.item()) * xb.size(0)
            total_correct += int((logits.argmax(dim=1) == yb).sum().item())
            total_n += xb.size(0)
            total_fr += float(fr.item()) * xb.size(0)

        tr_loss = float(total_loss) / max(total_n, 1)
        tr_acc = float(total_correct) / max(total_n, 1)
        tr_fr = float(total_fr) / max(total_n, 1)
        te = eval_model(hybrid_model, test_loader, device, args)

        print(f"[DSPR-FT] Epoch {epoch:02d} | eta_eff {eta_eff:.4e} | train loss {tr_loss:.4f} acc {tr_acc:.4f} fr_hid {tr_fr:.4f} | test loss {te['loss']:.4f} acc {te['acc']:.4f}")

        if te["acc"] > best_acc:
            best_acc = te["acc"]
            best_state = {
                "backbone": hybrid_model.backbone.state_dict(),
                "head_z_pos": hybrid_model.head.z_pos.detach().cpu(),
                "head_z_neg": hybrid_model.head.z_neg.detach().cpu(),
                "head_li_out": hybrid_model.head.li_out.state_dict(),
            }

    hybrid_model.backbone.load_state_dict(best_state["backbone"])
    hybrid_model.head.z_pos.data.copy_(best_state["head_z_pos"].to(device))
    hybrid_model.head.z_neg.data.copy_(best_state["head_z_neg"].to(device))
    hybrid_model.head.li_out.load_state_dict(best_state["head_li_out"])
    return best_acc, best_state


def eta_schedule(epoch: int, eta_net: float, warmup_epochs: int) -> float:
    if warmup_epochs <= 0:
        return eta_net
    alpha = min(1.0, epoch / float(warmup_epochs))
    return float(eta_net) * alpha


def run_hybrid_experiment(dataset_name: str, dspr: DSPRParams, args):
    device = pick_device(args.device)
    train_ds, n_out = get_dataset(dataset_name, args.data_dir, train=True)
    test_ds, _ = get_dataset(dataset_name, args.data_dir, train=False)

    train_ds = subset_dataset(train_ds, args.train_subset)
    test_ds = subset_dataset(test_ds, args.test_subset)

    train_loader = make_loader(train_ds, args.batch_size, train=True)
    test_loader = make_loader(test_ds, args.batch_size, train=False)

    backbone = ConvSNNBackbone(
        feature_dim=args.feature_dim,
        beta_h=args.beta_h,
        thr_h=args.thr_h,
        beta_feat=args.beta_feat,
        spike_slope=args.spike_slope,
        syn_scale_conv1=args.syn_scale_conv1,
        syn_scale_conv2=args.syn_scale_conv2,
        syn_scale_fc1=args.syn_scale_fc1,
    ).to(device)

    cont_head = ContinuousHead(args.feature_dim, n_out).to(device)
    cont_model = ContinuousBackboneModel(backbone, cont_head).to(device)

    ideal_acc, ideal_state = pretrain_continuous(cont_model, train_loader, test_loader, device, args)

    mapped_head = DSPRHead(
        feature_dim=args.feature_dim,
        n_out=n_out,
        dspr=dspr,
        wmax=args.head_wmax,
        init_mode=args.init_mode,
        init_spread=args.init_spread,
        history_lambda=args.history_lambda,
        tau_h=args.tau_h,
        history_increment=args.history_increment,
        noise_on=args.noise_on,
        leak_dt=args.leak_dt,
        delta_z_cap=args.delta_z_cap,
        li_beta_init=args.beta_out,
    ).to(device)
    with torch.no_grad():
        mapped_head.li_out.load_state_dict(cont_head.li_out.state_dict())
        mapped_head.initialize_from_continuous_weight(cont_head.fc.weight.detach())

    naive_model = HybridDSPRModel(backbone, mapped_head).to(device)
    naive_eval = eval_model(naive_model, test_loader, device, args)
    naive_acc = naive_eval["acc"]

    print("\n=== Phase 2: naive mapped DSPR head evaluation ===")
    print(f"[NAIVE MAP] test loss {naive_eval['loss']:.4f} acc {naive_eval['acc']:.4f}")

    ft_backbone = ConvSNNBackbone(
        feature_dim=args.feature_dim,
        beta_h=args.beta_h,
        thr_h=args.thr_h,
        beta_feat=args.beta_feat,
        spike_slope=args.spike_slope,
        syn_scale_conv1=args.syn_scale_conv1,
        syn_scale_conv2=args.syn_scale_conv2,
        syn_scale_fc1=args.syn_scale_fc1,
    ).to(device)
    ft_backbone.load_state_dict(ideal_state["backbone"])

    dspr_head = DSPRHead(
        feature_dim=args.feature_dim,
        n_out=n_out,
        dspr=dspr,
        wmax=args.head_wmax,
        init_mode=args.init_mode,
        init_spread=args.init_spread,
        history_lambda=args.history_lambda,
        tau_h=args.tau_h,
        history_increment=args.history_increment,
        noise_on=args.noise_on,
        leak_dt=args.leak_dt,
        delta_z_cap=args.delta_z_cap,
        li_beta_init=args.beta_out,
    ).to(device)
    with torch.no_grad():
        dspr_head.li_out.load_state_dict(cont_head.li_out.state_dict())
        dspr_head.initialize_from_continuous_weight(cont_head.fc.weight.detach())

    hybrid_model = HybridDSPRModel(ft_backbone, dspr_head).to(device)
    dspr_acc, dspr_state = finetune_dspr_head(hybrid_model, train_loader, test_loader, device, args)

    transfer_loss_naive = ideal_acc - naive_acc
    transfer_loss_dspr = ideal_acc - dspr_acc
    recovered = dspr_acc - naive_acc

    summary = {
        "dataset": dataset_name,
        "ideal_acc": ideal_acc,
        "naive_acc": naive_acc,
        "dspr_acc": dspr_acc,
        "transfer_loss_naive": transfer_loss_naive,
        "transfer_loss_dspr": transfer_loss_dspr,
        "recovered_acc": recovered,
    }

    save_path = f"{args.save_prefix}_{dataset_name.replace('-', '_')}_hybrid_head_dspr.pt"
    torch.save(
        {
            "summary": summary,
            "args": vars(args),
            "dspr": dspr.__dict__,
            "ideal_state": ideal_state,
            "naive_eval": naive_eval,
            "dspr_state": dspr_state,
        },
        save_path,
    )

    print("\n=== Hybrid transfer comparison ===")
    print(f"Ideal continuous acc      : {ideal_acc:.4f}")
    print(f"Naive mapped DSPR acc     : {naive_acc:.4f}")
    print(f"DSPR-aware fine-tuned acc : {dspr_acc:.4f}")
    print(f"Naive transfer loss       : {transfer_loss_naive:.4f}")
    print(f"DSPR transfer loss        : {transfer_loss_dspr:.4f}")
    print(f"Recovered accuracy        : {recovered:.4f}")
    print(f"Saved experiment to       : {save_path}")
    return summary, save_path


def load_raw_tsv(tsv_path: str) -> pd.DataFrame:
    df = pd.read_csv(tsv_path, sep="\t", engine="python")
    required = {"Time,s", "VOLTAGE", "CURRENT"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{tsv_path}: missing columns {missing}")
    return df


def extract_pulse_end_currents(df: pd.DataFrame, v_prog: float) -> np.ndarray:
    v = df["VOLTAGE"].to_numpy(dtype=np.float64)
    i = df["CURRENT"].to_numpy(dtype=np.float64)
    is_high = v >= 0.5 * float(v_prog)
    if not np.any(is_high):
        return np.array([], dtype=np.float64)
    rising = np.zeros_like(is_high, dtype=np.int32)
    rising[0] = int(is_high[0])
    rising[1:] = ((~is_high[:-1]) & (is_high[1:])).astype(np.int32)
    pulse_id = np.cumsum(rising)
    out = []
    for pid in np.unique(pulse_id[is_high]):
        idx = np.where((pulse_id == pid) & is_high)[0]
        out.append(i[idx[-1]])
    return np.asarray(out, dtype=np.float64)


def compare_topologies(paths: Dict[str, str], v_prog: float = 5.0):
    print("\n=== Auxiliary topology comparison (positive-only pulse-end currents) ===")
    for name, pth in paths.items():
        df = load_raw_tsv(pth)
        end_i = extract_pulse_end_currents(df, v_prog=v_prog)
        if end_i.size == 0:
            print(f"{name}: no high pulses found")
            continue
        mean_last = end_i[-10:].mean() if end_i.size >= 10 else end_i.mean()
        print(f"{name}: pulses={end_i.size}, first={end_i[0]:.4e} A, last={end_i[-1]:.4e} A, mean_last10={mean_last:.4e} A")


def parse_args():
    ap = argparse.ArgumentParser(description="Hybrid experiment: strong continuous backbone + naive mapping vs DSPR-aware head fine-tuning")
    ap.add_argument("--protocol_xlsx", type=str, required=True)
    ap.add_argument("--sheet_name", default="0")
    ap.add_argument("--datasets", type=str, default="FashionMNIST")
    ap.add_argument("--data_dir", type=str, default="./data")
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--train_subset", type=int, default=0)
    ap.add_argument("--test_subset", type=int, default=0)

    ap.add_argument("--T", type=int, default=20)
    ap.add_argument("--encoding", type=str, default="rate", choices=["rate", "latency"])
    ap.add_argument("--rate_scale", type=float, default=0.25)
    ap.add_argument("--latency_thr", type=float, default=0.15)
    ap.add_argument("--mem_logits", type=str, default="mean", choices=["sum", "mean", "max", "last"])
    ap.add_argument("--temporal_loss_weight", type=float, default=0.10)

    ap.add_argument("--feature_dim", type=int, default=256)
    ap.add_argument("--beta_h", type=float, default=0.90)
    ap.add_argument("--beta_feat", type=float, default=0.90)
    ap.add_argument("--beta_out", type=float, default=0.85)
    ap.add_argument("--thr_h", type=float, default=0.70)
    ap.add_argument("--spike_slope", type=float, default=12.0)
    ap.add_argument("--syn_scale_conv1", type=float, default=0.60)
    ap.add_argument("--syn_scale_conv2", type=float, default=0.45)
    ap.add_argument("--syn_scale_fc1", type=float, default=0.28)

    ap.add_argument("--pretrain_epochs", type=int, default=15)
    ap.add_argument("--pretrain_lr", type=float, default=1e-3)

    ap.add_argument("--finetune_epochs", type=int, default=10)
    ap.add_argument("--finetune_aux_lr", type=float, default=3e-4)
    ap.add_argument("--eta_net", type=float, default=2e-3)
    ap.add_argument("--eta_warmup_epochs", type=int, default=5)
    ap.add_argument("--head_wmax", type=float, default=1.0)
    ap.add_argument("--delta_z_cap", type=float, default=0.03)
    ap.add_argument("--noise_on", action="store_true")
    ap.add_argument("--leak_dt", type=float, default=0.0)
    ap.add_argument("--history_lambda", type=float, default=0.0)
    ap.add_argument("--tau_h", type=float, default=1.0)
    ap.add_argument("--history_increment", type=float, default=1.0)
    ap.add_argument("--init_mode", type=str, default="mid", choices=["low", "mid", "high"])
    ap.add_argument("--init_spread", type=float, default=0.03)

    ap.add_argument("--fr_lambda", type=float, default=1.0)
    ap.add_argument("--fr_target", type=float, default=0.10)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--weight_decay", type=float, default=1e-5)

    ap.add_argument("--raw_1m", type=str, default=None)
    ap.add_argument("--raw_1p1m", type=str, default=None)
    ap.add_argument("--raw_2p1m", type=str, default=None)
    ap.add_argument("--raw_3m", type=str, default=None)
    ap.add_argument("--save_prefix", type=str, default="gan_dspr_snn")
    return ap.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    if args.device == "auto" and torch.backends.mps.is_available():
        print("[INFO] MPS detected. For stable comparisons, --device cpu is preferred.")

    dspr, _ = fit_dspr_from_xlsx(args.protocol_xlsx, sheet_name=parse_sheet_name(args.sheet_name))
    print("Fitted DSPR parameters from protocol file:")
    print(dspr)

    topo_paths = {}
    if args.raw_1m: topo_paths["1m"] = args.raw_1m
    if args.raw_1p1m: topo_paths["1+1m"] = args.raw_1p1m
    if args.raw_2p1m: topo_paths["2+1m"] = args.raw_2p1m
    if args.raw_3m: topo_paths["3m"] = args.raw_3m
    if topo_paths:
        try:
            compare_topologies(topo_paths, v_prog=5.0)
        except Exception as e:
            print(f"[WARNING] Topology comparison failed: {e}")

    datasets_list = [d.strip() for d in args.datasets.split(",") if d.strip()]
    all_results = {}
    for dname in datasets_list:
        try:
            summary, save_path = run_hybrid_experiment(dname, dspr, args)
            all_results[dname] = {"summary": summary, "artifact": save_path}
        except Exception as e:
            print(f"\n[WARNING] Skipping dataset {dname} because of error: {e}")
            all_results[dname] = {"summary": None, "artifact": f"FAILED: {e}"}

    print("\n=== Final hybrid results ===")
    for k, v in all_results.items():
        if v["summary"] is None:
            print(f"{k}: FAILED | {v['artifact']}")
        else:
            s = v["summary"]
            print(f"{k}: ideal={s['ideal_acc']:.4f} | naive={s['naive_acc']:.4f} | dspr={s['dspr_acc']:.4f} | recovered={s['recovered_acc']:.4f}")


if __name__ == "__main__":
    main()
