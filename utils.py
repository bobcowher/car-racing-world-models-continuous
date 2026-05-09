from __future__ import annotations
import numpy as np
import cv2
import math
import torch
import torch.nn as nn


def _to_frames(obs, num_frames: int) -> list[np.ndarray]:
    """Convert any supported obs shape to a list of (H, W) float32 frames."""
    if hasattr(obs, 'numpy'):
        obs = obs.numpy()

    obs = obs.astype(np.float32)

    if obs.ndim == 4:
        obs = obs[0]

    if obs.ndim == 3 and obs.shape[0] == num_frames:
        return [obs[i] for i in range(num_frames)]

    if obs.ndim == 3:
        if obs.shape[0] < obs.shape[-1]:
            obs = obs.transpose(1, 2, 0)
        if obs.shape[-1] in (1, 3, 4):
            obs = obs[:, :, 0]

    h = obs.shape[0] // num_frames
    return [obs[i * h:(i + 1) * h, :] for i in range(num_frames)]


def _labeled_row(label: str, frames: list[np.ndarray], label_height: int = 20) -> np.ndarray:
    """Tile frames side-by-side and prepend a label bar."""
    frames_u8 = [(np.clip(f, 0.0, 1.0) * 255).astype(np.uint8) for f in frames]
    tiled = np.concatenate(frames_u8, axis=1)

    bar = np.zeros((label_height, tiled.shape[1]), dtype=np.uint8)
    cv2.putText(bar, label, (4, label_height - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,), 1, cv2.LINE_AA)

    return np.concatenate([bar, tiled], axis=0)


def display_stacked_obs(entries: list[tuple[str, object]], filename: str, num_frames: int = 4):
    """Save one or more labeled stacked observations to an image file.

    Args:
        entries:   List of (label, obs) pairs. obs can be (B,C,H,W), (C,H,W), or (H*C,W).
        filename:  Output path (e.g. 'debug_pred.png')
        num_frames: Number of frames per stacked observation
    """
    rows = []
    for label, obs in entries:
        frames = _to_frames(obs, num_frames)
        rows.append(_labeled_row(label, frames))

    cv2.imwrite(filename, np.concatenate(rows, axis=0))

def create_log_gaussian(mean, log_std, t):
    quadratic = -((0.5 * (t - mean) / (log_std.exp())).pow(2))
    l = mean.shape
    log_z = log_std
    z = l[-1] * math.log(2 * math.pi)
    log_p = quadratic.sum(dim=-1) - log_z.sum(dim=-1) - 0.5 * z
    return log_p

def logsumexp(inputs, dim=None, keepdim=False):
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)
