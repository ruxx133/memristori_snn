
import argparse
import json
import math
import random
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
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


def parse_seed_list(seed_str: str) -> List[int]:
    return [int(s.strip()) for s in seed_str.split(",") if s.strip()]


def subset_dataset(ds, subset: int):
    if subset is None or subset <= 0 or subset >= len(ds):
        return ds
    return torch.utils.data.Subset(ds, range(subset))


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
    df = pd.read_excel(xlsx_path, sheet_name=0 if sheet_name is None else sheet_name, engine="openpyxl")
    if isinstance(df, dict):
        df = next(iter(df.values()))
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
                sid=int(sid), start_idx=int(idx[0]), end_idx=int(idx[-1]), voltage=voltage,
                t0=float(t[idx[0]]), t1=float(t[idx[-1]]), duration=duration,
                current_median=current_median, current_last=current_last, n=len(idx), note=note, kind=kind
            )
        )
    return segments


def read_conductance_from_segment(seg: Segment) -> float:
    return seg.current_median / float(seg.voltage)


def robust_plateaus(read_G: np.ndarray) -> Tuple[float, float]:
    G_low = float(np.min(read_G))
    G_high = float(np.max(read_G))
    if G_high <= G_low:
        raise ValueError("Degenerate read conductance range.")
    return G_low, G_high


def normalize_G(G: float, G_low: float, G_high: float) -> float:
    return float(np.clip((G - G_low) / (G_high - G_low + 1e-12), 0.0, 1.0))


def estimate_kappas(segments: List[Segment], G_low: float, G_high: float) -> Tuple[float, float, float, float]:
    set_rates, reset_rates, set_res, reset_res = [], [], [], []
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

    if not set_rates or not reset_rates:
        raise ValueError("Could not estimate kappas from protocol.")

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


def extract_measured_levels(segments: List[Segment], G_low: float, G_high: float, round_decimals: int = 4) -> np.ndarray:
    read_segments = [s for s in segments if s.kind == "read" and abs(s.voltage) > 1e-12]
    zs = np.array([normalize_G(read_conductance_from_segment(s), G_low, G_high) for s in read_segments], dtype=np.float32)
    levels = np.unique(np.round(zs, round_decimals))
    levels = np.clip(levels, 0.0, 1.0)
    levels = np.sort(levels)
    if len(levels) < 4:
        levels = np.linspace(0.0, 1.0, 8, dtype=np.float32)
    return levels.astype(np.float32)


def fit_dspr_from_xlsx(xlsx_path: str, sheet_name: Optional[str] = None) -> Tuple[DSPRParams, List[Segment], np.ndarray]:
    df = load_protocol_xlsx(xlsx_path, sheet_name=sheet_name)
    segments = segment_protocol(df)
    read_segments = [s for s in segments if s.kind == "read" and abs(s.voltage) > 1e-12]
    read_G = np.array([read_conductance_from_segment(s) for s in read_segments], dtype=np.float64)
    G_low, G_high = robust_plateaus(read_G)
    kappa_plus, kappa_zero, sigma_plus, sigma_zero = estimate_kappas(segments, G_low, G_high)
    V_read_values = [s.voltage for s in read_segments]
    V_set_values = [s.voltage for s in segments if s.kind == "set"]
    V_reset_values = [s.voltage for s in segments if s.kind == "reset"]
    params = DSPRParams(
        G_low=G_low, G_high=G_high, kappa_plus=kappa_plus, kappa_zero=kappa_zero,
        sigma_plus=sigma_plus, sigma_zero=sigma_zero,
        V_read=float(np.median(V_read_values)) if V_read_values else 1.0,
        V_set=float(np.median(V_set_values)) if V_set_values else 5.0,
        V_reset=float(np.median(V_reset_values)) if V_reset_values else 0.0
    )
    levels = extract_measured_levels(segments, G_low, G_high)
    return params, segments, levels


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
    if name_l == "fashionmnist":
        return datasets.FashionMNIST(root=data_dir, train=train, download=True, transform=tfm), 10
    if name_l == "emnist-balanced":
        return datasets.EMNIST(root=data_dir, split="balanced", train=train, download=True, transform=tfm), 47
    raise ValueError(f"Unsupported dataset: {name}")


def safe_collate(batch):
    xs, ys = zip(*batch)
    xs_out = [x if isinstance(x, torch.Tensor) else TF.to_tensor(x) for x in xs]
    ys_out = [int(y.item()) if isinstance(y, torch.Tensor) else int(y) for y in ys]
    return torch.stack(xs_out, dim=0), torch.tensor(ys_out, dtype=torch.long)


def make_loader(ds, batch_size: int, train: bool):
    return DataLoader(ds, batch_size=batch_size, shuffle=train, drop_last=train, num_workers=0,
                      pin_memory=False, persistent_workers=False, collate_fn=safe_collate)


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


class ContinuousBackbone(nn.Module):
    def __init__(self, feature_dim=128, beta_h=0.90, thr_h=0.70, spike_slope=12.0,
                 syn_scale_conv1=0.70, syn_scale_conv2=0.55, syn_scale_fc1=0.35):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        self.fc1 = nn.Linear(64 * 7 * 7, feature_dim, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm1d(feature_dim)
        self.lif1 = LearnableLIFCell(beta_init=beta_h, threshold=thr_h, spike_slope=spike_slope)
        self.lif2 = LearnableLIFCell(beta_init=beta_h, threshold=thr_h, spike_slope=spike_slope)
        self.lif3 = LearnableLIFCell(beta_init=beta_h, threshold=thr_h, spike_slope=spike_slope)
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
        feat_rec, fr_accum = [], 0.0
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
        return torch.stack(feat_rec), fr_accum / (3.0 * T)


class ContinuousClassifier(nn.Module):
    def __init__(self, feature_dim, hidden_dim, n_out, beta_h=0.90, beta_out=0.85, thr_h=0.70,
                 spike_slope=12.0, syn_scale_fc2=0.25, syn_scale_fc3=0.18):
        super().__init__()
        self.fc2 = nn.Linear(feature_dim, hidden_dim, bias=False)
        self.bn4 = nn.BatchNorm1d(hidden_dim)
        self.lif4 = LearnableLIFCell(beta_init=beta_h, threshold=thr_h, spike_slope=spike_slope)
        self.fc3 = nn.Linear(hidden_dim, n_out, bias=False)
        self.li_out = LearnableLICell(beta_init=beta_out)
        self.syn_scale_fc2 = float(syn_scale_fc2)
        self.syn_scale_fc3 = float(syn_scale_fc3)
        self.hidden_dim = int(hidden_dim)
        self.n_out = int(n_out)

    def forward(self, feat_rec: torch.Tensor):
        T, B, _ = feat_rec.shape
        mem4 = torch.zeros(B, self.hidden_dim, device=feat_rec.device)
        mem_out = torch.zeros(B, self.n_out, device=feat_rec.device)
        out_rec, fr_accum = [], 0.0
        for t in range(T):
            cur4 = self.bn4(self.syn_scale_fc2 * self.fc2(feat_rec[t]))
            spk4, mem4 = self.lif4(cur4, mem4)
            cur5 = self.syn_scale_fc3 * self.fc3(spk4)
            mem_out = self.li_out(cur5, mem_out)
            out_rec.append(mem_out)
            fr_accum = fr_accum + spk4.mean()
        return torch.stack(out_rec), fr_accum / T


class ContinuousFullModel(nn.Module):
    def __init__(self, backbone, classifier):
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier

    def forward(self, spk_in):
        feat_rec, fr_back = self.backbone(spk_in)
        out_rec, fr_head = self.classifier(feat_rec)
        return out_rec, 0.75 * fr_back + 0.25 * fr_head


class DSPRQuantizedLinear(nn.Module):
    def __init__(self, in_features, out_features, levels, dspr, init_mode="mid", init_spread=0.03,
                 delta_z_cap=0.03, history_lambda=0.0, tau_h=1.0, history_increment=1.0,
                 programming_noise_std=0.02, read_noise_std=0.015, leak_dt=0.0, weight_scale=1.0):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.register_buffer("levels", torch.tensor(np.asarray(levels, dtype=np.float32)))
        self.kappa_plus = float(dspr.kappa_plus)
        self.kappa_zero = float(dspr.kappa_zero)
        self.delta_z_cap = float(delta_z_cap)
        self.history_lambda = float(history_lambda)
        self.tau_h = float(tau_h)
        self.history_increment = float(history_increment)
        self.programming_noise_std = float(programming_noise_std)
        self.read_noise_std = float(read_noise_std)
        self.leak_dt = float(leak_dt)
        self.eps = 1e-8
        self.register_buffer("weight_scale", torch.tensor(float(weight_scale), dtype=torch.float32))

        if init_mode == "low":
            mean_pos, mean_neg = 0.20, 0.10
        elif init_mode == "high":
            mean_pos, mean_neg = 0.75, 0.25
        else:
            mean_pos, mean_neg = 0.55, 0.45

        zpos = torch.empty(out_features, in_features).normal_(mean=mean_pos, std=init_spread).clamp_(0.0, 1.0)
        zneg = torch.empty(out_features, in_features).normal_(mean=mean_neg, std=init_spread).clamp_(0.0, 1.0)
        self.z_pos = nn.Parameter(zpos)
        self.z_neg = nn.Parameter(zneg)
        self.register_buffer("h_pos", torch.zeros(out_features, in_features))
        self.register_buffer("h_neg", torch.zeros(out_features, in_features))

    def set_weight_scale(self, scale: float):
        with torch.no_grad():
            self.weight_scale.fill_(float(scale))

    def _nearest_levels(self, z: torch.Tensor) -> torch.Tensor:
        diff = torch.abs(z.unsqueeze(-1) - self.levels.view(*([1] * z.ndim), -1))
        idx = diff.argmin(dim=-1)
        return self.levels[idx]

    def _quantize_ste(self, z: torch.Tensor) -> torch.Tensor:
        zq = self._nearest_levels(z)
        return zq.detach() + (z - z.detach())

    def _read_state(self, z: torch.Tensor, add_noise: bool) -> torch.Tensor:
        zq = self._quantize_ste(z)
        if add_noise and self.read_noise_std > 0.0:
            zq = zq + self.read_noise_std * torch.randn_like(zq)
        return zq.clamp(0.0, 1.0)

    def effective_weight(self, add_noise: bool) -> torch.Tensor:
        zpos = self._read_state(self.z_pos, add_noise=add_noise)
        zneg = self._read_state(self.z_neg, add_noise=add_noise)
        return self.weight_scale * (zpos - zneg)

    def forward(self, x: torch.Tensor, add_noise: bool = True) -> torch.Tensor:
        return x @ self.effective_weight(add_noise=add_noise).t()

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
        if self.programming_noise_std > 0.0:
            z_new = z_new + self.programming_noise_std * torch.randn_like(z_new)
        z_new = self._nearest_levels(z_new.clamp(0.0, 1.0))
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
        z_new = self._nearest_levels(z_new.clamp(0.0, 1.0))
        if self.history_lambda != 0.0 and self.tau_h > 0:
            h_new = torch.exp(torch.full_like(h, -self.leak_dt / self.tau_h)) * h
        else:
            h_new = h
        return z_new, h_new

    @torch.no_grad()
    def dspr_step(self, eta_net: float) -> None:
        if self.z_pos.grad is None or self.z_neg.grad is None:
            return

        scale = float(self.weight_scale.item())
        grad_w = 0.5 * (self.z_pos.grad / max(scale, self.eps) - self.z_neg.grad / max(scale, self.eps))
        desired_dw = -float(eta_net) * grad_w
        desired_dz = (desired_dw.abs() / max(scale, self.eps)).clamp(min=0.0, max=self.delta_z_cap)

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
        self.z_pos.copy_(zpl)
        self.z_neg.copy_(znl)
        self.h_pos.copy_(hpl)
        self.h_neg.copy_(hnl)
        self.z_pos.grad.zero_()
        self.z_neg.grad.zero_()

    @torch.no_grad()
    def initialize_from_weight(self, W: torch.Tensor):
        scale = float(W.abs().max().item())
        if scale < 1e-8:
            scale = 1.0
        self.set_weight_scale(scale)
        Wn = torch.clamp(W / scale, -1.0, 1.0)
        zpos = torch.clamp(Wn, min=0.0)
        zneg = torch.clamp(-Wn, min=0.0)
        self.z_pos.copy_(self._nearest_levels(zpos))
        self.z_neg.copy_(self._nearest_levels(zneg))
        self.h_pos.zero_()
        self.h_neg.zero_()


class DSPRTwoLayerClassifier(nn.Module):
    def __init__(self, feature_dim, hidden_dim, n_out, levels, dspr, beta_h=0.90, beta_out=0.85, thr_h=0.70,
                 spike_slope=12.0, syn_scale_fc2=0.25, syn_scale_fc3=0.18, init_mode="mid", init_spread=0.03,
                 delta_z_cap=0.03, history_lambda=0.0, tau_h=1.0, history_increment=1.0,
                 programming_noise_std=0.02, read_noise_std=0.015, leak_dt=0.0):
        super().__init__()
        self.fc2 = DSPRQuantizedLinear(feature_dim, hidden_dim, levels, dspr, init_mode=init_mode,
                                       init_spread=init_spread, delta_z_cap=delta_z_cap, history_lambda=history_lambda,
                                       tau_h=tau_h, history_increment=history_increment,
                                       programming_noise_std=programming_noise_std, read_noise_std=read_noise_std,
                                       leak_dt=leak_dt)
        self.bn4 = nn.BatchNorm1d(hidden_dim)
        self.lif4 = LearnableLIFCell(beta_init=beta_h, threshold=thr_h, spike_slope=spike_slope)
        self.fc3 = DSPRQuantizedLinear(hidden_dim, n_out, levels, dspr, init_mode=init_mode,
                                       init_spread=init_spread, delta_z_cap=delta_z_cap, history_lambda=history_lambda,
                                       tau_h=tau_h, history_increment=history_increment,
                                       programming_noise_std=programming_noise_std, read_noise_std=read_noise_std,
                                       leak_dt=leak_dt)
        self.li_out = LearnableLICell(beta_init=beta_out)
        self.syn_scale_fc2 = float(syn_scale_fc2)
        self.syn_scale_fc3 = float(syn_scale_fc3)
        self.hidden_dim = int(hidden_dim)
        self.n_out = int(n_out)

    def initialize_from_continuous(self, cont_classifier: ContinuousClassifier):
        with torch.no_grad():
            self.fc2.initialize_from_weight(cont_classifier.fc2.weight.detach())
            self.fc3.initialize_from_weight(cont_classifier.fc3.weight.detach())
            self.bn4.load_state_dict(cont_classifier.bn4.state_dict())
            self.lif4.rho.data.copy_(cont_classifier.lif4.rho.data)
            self.li_out.load_state_dict(cont_classifier.li_out.state_dict())

    def forward(self, feat_rec: torch.Tensor, add_noise: bool = True):
        T, B, _ = feat_rec.shape
        mem4 = torch.zeros(B, self.hidden_dim, device=feat_rec.device)
        mem_out = torch.zeros(B, self.n_out, device=feat_rec.device)
        out_rec, fr_accum = [], 0.0
        for t in range(T):
            cur4 = self.bn4(self.syn_scale_fc2 * self.fc2(feat_rec[t], add_noise=add_noise))
            spk4, mem4 = self.lif4(cur4, mem4)
            cur5 = self.syn_scale_fc3 * self.fc3(spk4, add_noise=add_noise)
            mem_out = self.li_out(cur5, mem_out)
            out_rec.append(mem_out)
            fr_accum = fr_accum + spk4.mean()
        return torch.stack(out_rec), fr_accum / T

    def dspr_step(self, eta_net: float):
        self.fc2.dspr_step(eta_net)
        self.fc3.dspr_step(eta_net)


class HybridTwoLayerModel(nn.Module):
    def __init__(self, backbone, classifier):
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier

    def forward(self, spk_in, add_noise: bool = True):
        feat_rec, fr_back = self.backbone(spk_in)
        out_rec, fr_head = self.classifier(feat_rec, add_noise=add_noise)
        return out_rec, 0.75 * fr_back + 0.25 * fr_head


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


def eta_schedule(epoch: int, eta_net: float, warmup_epochs: int) -> float:
    if warmup_epochs <= 0:
        return eta_net
    alpha = min(1.0, epoch / float(warmup_epochs))
    return float(eta_net) * alpha


@torch.no_grad()
def eval_model(model, loader, device, args, hardware_noise: bool):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_n = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = torch.as_tensor(yb, device=device, dtype=torch.long)
        spk_in = encode_rate(xb, args.T, args.rate_scale) if args.encoding == "rate" else encode_latency(xb, args.T, args.latency_thr)
        if isinstance(model, HybridTwoLayerModel):
            mem_rec, _ = model(spk_in, add_noise=hardware_noise)
        else:
            mem_rec, _ = model(spk_in)
        loss, logits = compute_losses(mem_rec, yb, args.mem_logits, args.temporal_loss_weight)
        total_loss += float(loss.item()) * xb.size(0)
        total_correct += int((logits.argmax(dim=1) == yb).sum().item())
        total_n += xb.size(0)
    return {"loss": float(total_loss) / max(total_n, 1), "acc": float(total_correct) / max(total_n, 1)}


def pretrain_continuous(model, train_loader, test_loader, device, args):
    params = list(model.parameters())
    opt = torch.optim.Adam(params, lr=args.pretrain_lr, weight_decay=args.weight_decay)
    best_acc = -1.0
    best_state = None
    print("\n=== Phase 1: continuous pretraining ===")
    for epoch in range(1, args.pretrain_epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_n = 0
        total_fr = 0.0
        printed_debug = False
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = torch.as_tensor(yb, device=device, dtype=torch.long)
            spk_in = encode_rate(xb, args.T, args.rate_scale) if args.encoding == "rate" else encode_latency(xb, args.T, args.latency_thr)
            opt.zero_grad(set_to_none=True)
            mem_rec, fr = model(spk_in)
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
        te = eval_model(model, test_loader, device, args, hardware_noise=False)
        print(f"[PRETRAIN] Epoch {epoch:02d} | train loss {tr_loss:.4f} acc {tr_acc:.4f} fr_hid {tr_fr:.4f} | test loss {te['loss']:.4f} acc {te['acc']:.4f}")
        if te["acc"] > best_acc:
            best_acc = te["acc"]
            best_state = {"backbone": model.backbone.state_dict(), "classifier": model.classifier.state_dict()}

    model.backbone.load_state_dict(best_state["backbone"])
    model.classifier.load_state_dict(best_state["classifier"])
    return best_acc, best_state


def finetune_dspr(model, train_loader, test_loader, device, args):
    for p in model.backbone.parameters():
        p.requires_grad = False
    for mod in [model.backbone.bn1, model.backbone.bn2, model.backbone.bn3,
                model.backbone.lif1, model.backbone.lif2, model.backbone.lif3,
                model.classifier.bn4, model.classifier.lif4, model.classifier.li_out]:
        for p in mod.parameters():
            p.requires_grad = True

    aux_params = []
    for name, p in model.named_parameters():
        if name.startswith("classifier.fc2.z_") or name.startswith("classifier.fc3.z_"):
            continue
        if p.requires_grad:
            aux_params.append(p)

    opt = torch.optim.Adam(aux_params, lr=args.finetune_aux_lr, weight_decay=args.weight_decay)
    best_acc = -1.0

    print("\n=== Phase 3: DSPR-aware two-layer fine-tuning ===")
    for epoch in range(1, args.finetune_epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_n = 0
        total_fr = 0.0
        eta_eff = eta_schedule(epoch, args.eta_net, args.eta_warmup_epochs)
        printed_debug = False

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = torch.as_tensor(yb, device=device, dtype=torch.long)
            spk_in = encode_rate(xb, args.T, args.rate_scale) if args.encoding == "rate" else encode_latency(xb, args.T, args.latency_thr)
            model.zero_grad(set_to_none=True)
            opt.zero_grad(set_to_none=True)
            mem_rec, fr = model(spk_in, add_noise=True)
            loss, logits = compute_losses(mem_rec, yb, args.mem_logits, args.temporal_loss_weight)
            if args.fr_lambda > 0.0:
                loss = loss + args.fr_lambda * (fr - args.fr_target) ** 2
            loss.backward()

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(aux_params + [
                    model.classifier.fc2.z_pos, model.classifier.fc2.z_neg,
                    model.classifier.fc3.z_pos, model.classifier.fc3.z_neg
                ], args.grad_clip)

            opt.step()
            model.classifier.dspr_step(eta_eff)

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
        te = eval_model(model, test_loader, device, args, hardware_noise=True)
        print(f"[DSPR-FT] Epoch {epoch:02d} | eta_eff {eta_eff:.4e} | train loss {tr_loss:.4f} acc {tr_acc:.4f} fr_hid {tr_fr:.4f} | test loss {te['loss']:.4f} acc {te['acc']:.4f}")
        best_acc = max(best_acc, te["acc"])
    return best_acc


def run_single_seed(dataset_name: str, dspr: DSPRParams, levels: np.ndarray, args, seed: int):
    set_seed(seed)
    device = pick_device(args.device)
    train_ds, n_out = get_dataset(dataset_name, args.data_dir, train=True)
    test_ds, _ = get_dataset(dataset_name, args.data_dir, train=False)
    train_ds = subset_dataset(train_ds, args.train_subset)
    test_ds = subset_dataset(test_ds, args.test_subset)
    train_loader = make_loader(train_ds, args.batch_size, train=True)
    test_loader = make_loader(test_ds, args.batch_size, train=False)

    backbone = ContinuousBackbone(feature_dim=args.feature_dim, beta_h=args.beta_h, thr_h=args.thr_h,
                                  spike_slope=args.spike_slope, syn_scale_conv1=args.syn_scale_conv1,
                                  syn_scale_conv2=args.syn_scale_conv2, syn_scale_fc1=args.syn_scale_fc1).to(device)
    cont_classifier = ContinuousClassifier(feature_dim=args.feature_dim, hidden_dim=args.head_hidden_dim, n_out=n_out,
                                           beta_h=args.beta_h, beta_out=args.beta_out, thr_h=args.thr_h,
                                           spike_slope=args.spike_slope, syn_scale_fc2=args.syn_scale_fc2,
                                           syn_scale_fc3=args.syn_scale_fc3).to(device)
    cont_model = ContinuousFullModel(backbone, cont_classifier).to(device)

    print(f"\n================ SEED {seed} | DATASET {dataset_name} ================\n")
    ideal_acc, ideal_state = pretrain_continuous(cont_model, train_loader, test_loader, device, args)

    mapped_classifier = DSPRTwoLayerClassifier(feature_dim=args.feature_dim, hidden_dim=args.head_hidden_dim, n_out=n_out,
                                               levels=levels, dspr=dspr, beta_h=args.beta_h, beta_out=args.beta_out,
                                               thr_h=args.thr_h, spike_slope=args.spike_slope,
                                               syn_scale_fc2=args.syn_scale_fc2, syn_scale_fc3=args.syn_scale_fc3,
                                               init_mode=args.init_mode, init_spread=args.init_spread,
                                               delta_z_cap=args.delta_z_cap, history_lambda=args.history_lambda,
                                               tau_h=args.tau_h, history_increment=args.history_increment,
                                               programming_noise_std=args.programming_noise_std,
                                               read_noise_std=args.read_noise_std, leak_dt=args.read_drift_dt).to(device)
    mapped_classifier.initialize_from_continuous(cont_classifier)
    naive_model = HybridTwoLayerModel(backbone, mapped_classifier).to(device)
    naive_eval = eval_model(naive_model, test_loader, device, args, hardware_noise=True)
    naive_acc = naive_eval["acc"]
    print("\n=== Phase 2: naive mapped DSPR last-two-layer evaluation ===")
    print(f"[NAIVE MAP] test loss {naive_eval['loss']:.4f} acc {naive_eval['acc']:.4f}")

    ft_backbone = ContinuousBackbone(feature_dim=args.feature_dim, beta_h=args.beta_h, thr_h=args.thr_h,
                                     spike_slope=args.spike_slope, syn_scale_conv1=args.syn_scale_conv1,
                                     syn_scale_conv2=args.syn_scale_conv2, syn_scale_fc1=args.syn_scale_fc1).to(device)
    ft_backbone.load_state_dict(ideal_state["backbone"])
    ft_classifier = DSPRTwoLayerClassifier(feature_dim=args.feature_dim, hidden_dim=args.head_hidden_dim, n_out=n_out,
                                           levels=levels, dspr=dspr, beta_h=args.beta_h, beta_out=args.beta_out,
                                           thr_h=args.thr_h, spike_slope=args.spike_slope,
                                           syn_scale_fc2=args.syn_scale_fc2, syn_scale_fc3=args.syn_scale_fc3,
                                           init_mode=args.init_mode, init_spread=args.init_spread,
                                           delta_z_cap=args.delta_z_cap, history_lambda=args.history_lambda,
                                           tau_h=args.tau_h, history_increment=args.history_increment,
                                           programming_noise_std=args.programming_noise_std,
                                           read_noise_std=args.read_noise_std, leak_dt=args.read_drift_dt).to(device)
    ft_classifier.initialize_from_continuous(cont_classifier)
    dspr_model = HybridTwoLayerModel(ft_backbone, ft_classifier).to(device)
    dspr_acc = finetune_dspr(dspr_model, train_loader, test_loader, device, args)

    summary = {
        "seed": seed,
        "dataset": dataset_name,
        "ideal_acc": ideal_acc,
        "naive_acc": naive_acc,
        "dspr_acc": dspr_acc,
        "transfer_loss_naive": ideal_acc - naive_acc,
        "transfer_loss_dspr": ideal_acc - dspr_acc,
        "recovered_acc": dspr_acc - naive_acc,
        "measured_levels": int(len(levels)),
    }
    print("\n=== Seed summary ===")
    print(json.dumps(summary, indent=2))
    return summary


def summarize_across_seeds(results: List[Dict]) -> Dict[str, float]:
    keys = ["ideal_acc", "naive_acc", "dspr_acc", "transfer_loss_naive", "transfer_loss_dspr", "recovered_acc"]
    out = {}
    for k in keys:
        vals = np.array([r[k] for r in results], dtype=np.float64)
        out[f"{k}_mean"] = float(vals.mean())
        out[f"{k}_std"] = float(vals.std(ddof=0))
    return out


def parse_args():
    ap = argparse.ArgumentParser(description="Harder hybrid DSPR experiment.")
    ap.add_argument("--protocol_xlsx", type=str, required=True)
    ap.add_argument("--sheet_name", default="0")
    ap.add_argument("--datasets", type=str, default="EMNIST-Balanced")
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--data_dir", type=str, default="./data")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--train_subset", type=int, default=0)
    ap.add_argument("--test_subset", type=int, default=0)

    ap.add_argument("--T", type=int, default=20)
    ap.add_argument("--encoding", type=str, default="rate", choices=["rate", "latency"])
    ap.add_argument("--rate_scale", type=float, default=0.25)
    ap.add_argument("--latency_thr", type=float, default=0.15)
    ap.add_argument("--mem_logits", type=str, default="mean", choices=["sum", "mean", "max", "last"])
    ap.add_argument("--temporal_loss_weight", type=float, default=0.10)

    ap.add_argument("--feature_dim", type=int, default=128)
    ap.add_argument("--head_hidden_dim", type=int, default=96)
    ap.add_argument("--beta_h", type=float, default=0.90)
    ap.add_argument("--beta_out", type=float, default=0.85)
    ap.add_argument("--thr_h", type=float, default=0.70)
    ap.add_argument("--spike_slope", type=float, default=12.0)
    ap.add_argument("--syn_scale_conv1", type=float, default=0.70)
    ap.add_argument("--syn_scale_conv2", type=float, default=0.55)
    ap.add_argument("--syn_scale_fc1", type=float, default=0.35)
    ap.add_argument("--syn_scale_fc2", type=float, default=0.25)
    ap.add_argument("--syn_scale_fc3", type=float, default=0.18)

    ap.add_argument("--pretrain_epochs", type=int, default=12)
    ap.add_argument("--pretrain_lr", type=float, default=1e-3)
    ap.add_argument("--finetune_epochs", type=int, default=10)
    ap.add_argument("--finetune_aux_lr", type=float, default=3e-4)
    ap.add_argument("--eta_net", type=float, default=2e-3)
    ap.add_argument("--eta_warmup_epochs", type=int, default=5)
    ap.add_argument("--delta_z_cap", type=float, default=0.03)

    ap.add_argument("--programming_noise_std", type=float, default=0.02)
    ap.add_argument("--read_noise_std", type=float, default=0.015)
    ap.add_argument("--read_drift_dt", type=float, default=0.0)

    ap.add_argument("--history_lambda", type=float, default=0.0)
    ap.add_argument("--tau_h", type=float, default=1.0)
    ap.add_argument("--history_increment", type=float, default=1.0)
    ap.add_argument("--init_mode", type=str, default="mid", choices=["low", "mid", "high"])
    ap.add_argument("--init_spread", type=float, default=0.03)

    ap.add_argument("--fr_lambda", type=float, default=1.0)
    ap.add_argument("--fr_target", type=float, default=0.10)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--weight_decay", type=float, default=1e-5)

    ap.add_argument("--save_prefix", type=str, default="gan_dspr_harder")
    return ap.parse_args()


def main():
    args = parse_args()
    dspr, segments, levels = fit_dspr_from_xlsx(args.protocol_xlsx, sheet_name=parse_sheet_name(args.sheet_name))
    print("Fitted DSPR parameters from protocol file:")
    print(dspr)
    print(f"Measured discrete levels extracted from protocol: {len(levels)}")
    print(f"Levels (first up to 12): {levels[:12]}")

    datasets_list = [d.strip() for d in args.datasets.split(",") if d.strip()]
    seeds = parse_seed_list(args.seeds)

    all_outputs = {}
    for dataset_name in datasets_list:
        dataset_results = []
        for seed in seeds:
            summary = run_single_seed(dataset_name, dspr, levels, args, seed)
            dataset_results.append(summary)
        aggregate = summarize_across_seeds(dataset_results)
        all_outputs[dataset_name] = {"per_seed": dataset_results, "aggregate": aggregate}
        print(f"\n=== Aggregate over seeds for {dataset_name} ===")
        for k, v in aggregate.items():
            print(f"{k}: {v:.4f}")

    out_path = Path(f"{args.save_prefix}_summary.json")
    out_path.write_text(json.dumps(all_outputs, indent=2), encoding="utf-8")
    print(f"\nSaved summary to {out_path.resolve()}")


if __name__ == "__main__":
    main()
