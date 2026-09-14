"""
experiments_utils.py

References:
Analysis derived from: Wu et al. (2025) - Adversarial imitation learning with deep attention network for swarm systems (https://doi.org/10.1007/s40747-024-01662-2)
Heat maps: https://matplotlib.org/stable/gallery/images_contours_and_fields/pcolor_demo.html
Heat maps: https://matplotlib.org/stable/users/explain/colors/colormapnorms.html

Wirtheim (2026) - Exploring Predator-Prey Dynamics from Videos using Generative Adversarial Imitation Learning

Note:
This thesis extends Wirtheim (2026)'s thesis, and this code is adapted from his implementation. 
"""

import os
import torch
import math
import re
import glob
import json
import numpy as np
import pandas as pd
from utils.sim_utils import *
from utils.eval_utils import *
from utils.train_utils import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.colors as colors
from scipy.optimize import linear_sum_assignment
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from typing import Any, Iterable, Mapping, Sequence


def compute_expert_data_ranges(expert_data):
    """
    Outputs the value ranges for x, y, vx, vy from expert data
    Needed for normalization and undistorted data plotting

    Input: expert data
    Output: value ranges for x, y, vx, vy
    """

    all_x = []
    all_y = []
    all_vx = []
    all_vy = []

    for video in expert_data.keys():
        # get data
        xs  = np.asarray(expert_data[video]["xs"])
        ys  = np.asarray(expert_data[video]["ys"])
        vxs = np.asarray(expert_data[video]["vxs"])
        vys = np.asarray(expert_data[video]["vys"])

        all_x.append(xs)
        all_y.append(ys)
        all_vx.append(vxs)
        all_vy.append(vys)

    # concatenate all data
    all_x  = np.concatenate(all_x)
    all_y  = np.concatenate(all_y)
    all_vx = np.concatenate(all_vx)
    all_vy = np.concatenate(all_vy)

    # compute min/max
    x_min, x_max = float(all_x.min()), float(all_x.max())
    y_min, y_max = float(all_y.min()), float(all_y.max())
    vx_min, vx_max = float(all_vx.min()), float(all_vx.max())
    vy_min, vy_max = float(all_vy.min()), float(all_vy.max())

    return (x_min, x_max, y_min, y_max, vx_min, vx_max, vy_min, vy_max)



def compute_expert_metrics(expert_data, num_agents):
    """
    Calculates swarm metrics from expert data

    Input: expert data, number of agents
    Output: polarization, angular momentum, degree of sparsity, distance to predator, escape alignment
    """

    polarizations = []
    angular_momenta = []
    sparsities = []
    distances_to_predator = []
    distance_nearest_prey = []
    escape_alignments = []

    # compute data ranges for normalization
    x_min, x_max, y_min, y_max, vx_min, vx_max, vy_min, vy_max = compute_expert_data_ranges(expert_data)

    for video in expert_data.keys():
        # get data
        xs  = np.asarray(expert_data[video]["xs"])
        ys  = np.asarray(expert_data[video]["ys"])
        vxs = np.asarray(expert_data[video]["vxs"])
        vys = np.asarray(expert_data[video]["vys"])

        # normalize data to [-1, 1]
        vxs = 2 * (vxs - vx_min) / (vx_max - vx_min) -1
        vys = 2 * (vys - vy_min) / (vy_max - vy_min) -1

        # reshape data
        time = len(xs) // num_agents
        xs  = xs.reshape(time, num_agents)
        ys  = ys.reshape(time, num_agents)
        vxs = vxs.reshape(time, num_agents)
        vys = vys.reshape(time, num_agents)

        for t in range(time):
            # get positions and velocities
            x_t  = xs[t]
            y_t  = ys[t]
            vx_t = vxs[t]
            vy_t = vys[t]

            # compute metrics
            polarizations.append(compute_polarization(vx_t, vy_t))
            angular_momenta.append(compute_angular_momentum(x_t, y_t, vx_t, vy_t))
            sparsities.append(degree_of_sparsity(x_t, y_t))
            distances_to_predator.append(distance_to_predator(x_t, y_t))
            distance_nearest_prey.append(pred_distance_to_nearest_prey(x_t, y_t))
            escape_alignments.append(escape_alignment(x_t, y_t, vx_t, vy_t))

    return {"polarization": polarizations,
            "angular_momentum": angular_momenta,
            "sparsity": sparsities,
            "distance_to_predator": distances_to_predator,
            "distance_nearest_prey": distance_nearest_prey,
            "escape_alignment": escape_alignments}



def minmax_norm(data, min=None, max=None):
    """
    Applies min-max normalization (for visualization purposes)

    Input: data
    Output: normalized data range [0, 1]
    """
    data = np.asarray(data, dtype=np.float32)
    if min is None or max is None:
        min, max = data.min(), data.max()
    return (data - min) / (max - min + 1e-8)


###############################
##### PURSUIT AND EVASION #####
###############################

def _as_float_tensor(value: Any, *, device: Any = None, dtype: Any = torch.float32) -> torch.Tensor:
    """
    Convert once while preserving an existing tensor's device by default.
    """

    if torch.is_tensor(value):
        target_device = value.device if device is None else device
        return value.to(device=target_device, dtype=dtype)
    return torch.as_tensor(value, device=device, dtype=dtype)


def _unit_vectors(vectors: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return vectors / vectors.norm(dim=-1, keepdim=True).clamp_min(eps)


def prepare_geometry_cache(
    data,
    prey_positions=None,
    predator_headings=None,
    prey_headings=None,
    *,
    d_source: float,
    time_chunk_size: int = 4096,
    device: Any = None,
    dtype: Any = torch.float32,
    include_prey_neighbors: bool = False,
):
    """
    Calculate reusable geometry for one clip/rollout.
    """

    if isinstance(data, Mapping):
        predator_positions = data["predator_positions"]
        prey_positions = data["prey_positions"]
        predator_headings = data.get("predator_headings", predator_headings)
        prey_headings = data.get("prey_headings", prey_headings)
    else:
        predator_positions = data
    if prey_positions is None:
        raise ValueError("prey_positions are required")
    if not np.isfinite(d_source) or d_source <= 0:
        raise ValueError("d_source must be a positive environment maximum distance")

    predator_positions = _as_float_tensor(predator_positions, device=device, dtype=dtype)
    prey_positions = _as_float_tensor(prey_positions, device=predator_positions.device, dtype=dtype)
    if predator_positions.ndim != 2 or predator_positions.shape[-1] != 2:
        raise ValueError("predator_positions must have shape [T, 2]")
    if prey_positions.ndim != 3 or prey_positions.shape[-1] != 2:
        raise ValueError("prey_positions must have shape [T, N, 2]")
    if predator_positions.shape[0] != prey_positions.shape[0]:
        raise ValueError("predator and prey positions must share T")

    displacement = prey_positions - predator_positions[:, None, :]
    distance = displacement.norm(dim=-1)
    nearest_distance, nearest_index = distance.min(dim=1)
    nearest_displacement = displacement.gather(
        1, nearest_index[:, None, None].expand(-1, 1, 2)
    ).squeeze(1)
    cache = {
        "predator_positions": predator_positions,
        "prey_positions": prey_positions,
        "predator_to_prey": displacement,
        "predator_to_prey_distance": distance,
        "nearest_prey_index": nearest_index,
        "nearest_prey_distance": nearest_distance,
        "nearest_prey_distance_norm": nearest_distance / float(d_source),
        "nearest_prey_direction": _unit_vectors(nearest_displacement),
    }
    if predator_headings is not None:
        heading = _as_float_tensor(predator_headings, device=predator_positions.device, dtype=dtype)
        if heading.shape != predator_positions.shape[:1]:
            raise ValueError("predator_headings must have shape [T]")
        cache["predator_headings"] = heading
        cache["predator_heading_vectors"] = torch.stack((heading.cos(), heading.sin()), dim=-1)
    if prey_headings is not None:
        heading = _as_float_tensor(prey_headings, device=predator_positions.device, dtype=dtype)
        if heading.shape != prey_positions.shape[:2]:
            raise ValueError("prey_headings must have shape [T, N]")
        cache["prey_headings"] = heading
        cache["prey_heading_vectors"] = torch.stack((heading.cos(), heading.sin()), dim=-1)
    return cache


def compute_predator_nearest_distance(geometry):
    """
    Return raw and environment-normalized nearest-prey distance.
    """

    return geometry["nearest_prey_distance"], geometry["nearest_prey_distance_norm"]


def compute_closing_speed(nearest_distance: Any, *, lag: int = 3) -> torch.Tensor:
    """
    Mean nearest-distance decrease per step over ``lag`` consecutive steps.
    """

    distance = _as_float_tensor(nearest_distance)
    if distance.ndim != 1:
        raise ValueError("nearest_distance must be one-dimensional")
    if lag < 1:
        raise ValueError("lag must be positive")
    if lag >= len(distance):
        return distance.new_empty(0)
    return (distance[:-lag] - distance[lag:]) / float(lag)


def compute_pursuit_alignment(geometry) -> torch.Tensor:
    """
    Predator heading alignment with the independently nearest prey.
    """

    if "predator_heading_vectors" not in geometry:
        raise ValueError("geometry does not contain predator headings")
    return (geometry["predator_heading_vectors"] * geometry["nearest_prey_direction"]).sum(dim=-1)


def compute_continuous_predator_response_tensor(geometry, *, lag: int = 1) -> torch.Tensor:
    """
    Fixed-threat-direction continuous response ``R`` for every prey.
    """

    if lag < 1:
        raise ValueError("lag must be positive")
    headings = geometry.get("prey_heading_vectors")
    if headings is None:
        raise ValueError("geometry does not contain prey headings")
    if lag >= headings.shape[0]:
        return headings.new_empty((0, headings.shape[1]))
    fixed_away = _unit_vectors(geometry["predator_to_prey"][:-lag])
    before = (headings[:-lag] * fixed_away).sum(dim=-1)
    after = (headings[lag:] * fixed_away).sum(dim=-1)
    return after - before


### reshaping for collab file ###

# settings
ARENA = 2160.0                       # px, area width used to scale the positions back to pixels
D_SOURCE = math.hypot(2160, 2160)    # maximum distance in the environment
CLOSING_LAG = 3                      # steps used for the closing speed
RESPONSE_LAG = 1                     # steps used for the prey reorientation

COLORS = {"GAIL": "#0072B2", "BC": "#D55E00", "Couzin": "#009E73", "Random": "#CC79A7"}


def headings_from_velocity(vx, vy):
    """
    Converts velocities into heading angles.

    Input: velocity components
    Output: heading angle in radians, NaN where the agent did not move
    """
    vx, vy = np.asarray(vx, float), np.asarray(vy, float)
    heading = np.arctan2(vy, vx)
    heading[np.hypot(vx, vy) < 1e-12] = np.nan
    return heading


def compute_pursuit_metrics(pred_pos, prey_pos, pred_head, prey_head):
    """
    Runs functions on one rollout or one attack clip

    Input: predator positions [T, 2], prey positions [T, N, 2] in px, headings in radians
    Output: pursuit alignment, closing speed, prey reorientation R, nearest prey distance
    """
    geometry = prepare_geometry_cache(pred_pos, prey_pos, pred_head, prey_head,
                                      d_source=D_SOURCE, include_prey_neighbors=False)
    raw_distance, _ = compute_predator_nearest_distance(geometry)
    closing = compute_closing_speed(raw_distance, lag=CLOSING_LAG)
    pursuit = compute_pursuit_alignment(geometry)
    response = compute_continuous_predator_response_tensor(geometry, lag=RESPONSE_LAG)
    return {"pursuit_alignment": pursuit.cpu().numpy(),
            "closing_speed": closing.cpu().numpy(),
            "reorientation": np.nanmean(response.cpu().numpy(), axis=1),
            "nearest_distance": raw_distance.cpu().numpy()}


def pursuit_metrics_from_simulation(metrics):
    """
    Computes pursuit alignment, closing speed and prey reorientation for one simulation

    Input: model metrics (run_env_simulation tuple or run_couzin_simulation list)
    Output: the three metrics as time series
    """
    # same unpacking as plot_pred_prey_metrics
    steps = metrics[0] if isinstance(metrics, tuple) else metrics

    # positions are stored scaled by the area width, so scale back to px
    xs = np.stack([np.asarray(m["xs"], float) for m in steps]) * ARENA
    ys = np.stack([np.asarray(m["ys"], float) for m in steps]) * ARENA
    vxs = np.stack([np.asarray(m["vxs"], float) for m in steps])
    vys = np.stack([np.asarray(m["vys"], float) for m in steps])

    positions = np.stack([xs, ys], axis=-1)      # [T, 1 + N, 2], index 0 = predator
    headings = headings_from_velocity(vxs, vys)  # [T, 1 + N]
    return compute_pursuit_metrics(positions[:, 0], positions[:, 1:], headings[:, 0], headings[:, 1:])


def safe_polarization(vx, vy):
    """
    Polarization for hand-labelled data (compute_polarization divides by zero for fish that do not move)

    Input: velocity components
    Output: polarization score of the prey
    """
    stacked_vs = np.stack([vx, vy], axis=1)
    norms = np.linalg.norm(stacked_vs, axis=1, keepdims=True)
    norm_vs = stacked_vs / (norms + 1e-8)
    return float(np.linalg.norm(norm_vs[1:].mean(axis=0)))


def load_attack_frames(path):
    """
    Reads one hand-labelled Label Studio attack file

    Input: path to a pred_attack_*.json file
    Output: list of (frame number, prey positions, predator position), coordinates in [0, 1]
    """
    d = json.load(open(path))
    frames = []
    for t in d:
        m = re.search(r'frame_(\d+)', t['data']['img'])
        if not m:
            continue
        prey, pred = [], None
        for r in t['annotations'][0]['result']:
            v = r['value']
            xy = np.array([v['x'], v['y']]) / 100.0
            if v['keypointlabels'][0] == 'Prey':
                prey.append(xy)
            elif v['keypointlabels'][0] == 'Predator':
                pred = xy
        if pred is None or not prey:
            continue
        frames.append((int(m.group(1)), np.array(prey), pred))
    frames.sort(key=lambda f: f[0])
    return frames


def compute_attack_expert_metrics(attack_dir):
    """
    Calculates all expert reference values from the hand-labelled attack clips

    Input: folder with pred_attack_*.json files
    Output: polarization, angular momentum, sparsity, distance to predator, nearest prey
            distance, escape alignment (in labelling units, as in compute_expert_metrics)
            and pursuit alignment, closing speed, reorientation, nearest distance (in px)
    """

    json_files = sorted(glob.glob(os.path.join(attack_dir, 'pred_attack_*.json')),
                        key=lambda p: int(re.search(r'(\d+)', os.path.basename(p)).group(1)))
    assert len(json_files) > 0, f"NO JSONs FOUND in {attack_dir}"
    print(f"  Found {len(json_files)} attack JSONs in {attack_dir}")

    metrics = {"polarization": [], "angular_momentum": [], "sparsity": [],
               "distance_to_predator": [], "distance_nearest_prey": [], "escape_alignment": [],
               "pursuit_alignment": [], "closing_speed": [], "reorientation": [], "nearest_distance": []}

    for f in json_files:
        frames = load_attack_frames(f)
        if len(frames) < 2:
            continue

        # match prey identities between frames (Hungarian matching)
        ordered, preds = [frames[0][1]], [frames[0][2]]
        for i in range(1, len(frames)):
            prev, cur = ordered[-1], frames[i][1]
            k = min(prev.shape[0], cur.shape[0])
            cost = np.linalg.norm(prev[:k, None, :] - cur[None, :k, :], axis=2)
            r, c = linear_sum_assignment(cost)
            o = cur[:k].copy()
            o[r] = cur[:k][c]
            ordered.append(o)
            preds.append(frames[i][2])

        # per-frame metrics, velocity = next frame - current frame
        for t in range(min(len(ordered), len(preds)) - 1):
            if ordered[t].shape[0] != ordered[t + 1].shape[0]:
                continue
            pp, pd = ordered[t], preds[t]
            pv, dv = ordered[t + 1] - ordered[t], preds[t + 1] - preds[t]
            xs = np.concatenate([[pd[0]], pp[:, 0]])
            ys = np.concatenate([[pd[1]], pp[:, 1]])
            vxs = np.concatenate([[dv[0]], pv[:, 0]])
            vys = np.concatenate([[dv[1]], pv[:, 1]])
            metrics["polarization"].append(safe_polarization(vxs, vys))
            metrics["angular_momentum"].append(compute_angular_momentum(xs, ys, vxs, vys))
            metrics["sparsity"].append(degree_of_sparsity(xs, ys))
            metrics["distance_to_predator"].append(distance_to_predator(xs, ys))
            metrics["distance_nearest_prey"].append(pred_distance_to_nearest_prey(xs, ys))
            metrics["escape_alignment"].append(escape_alignment(xs, ys, vxs, vys))

        # split the clip into stretches in which no prey appears or disappears
        start = 0
        for t in range(1, len(ordered) + 1):
            if t < len(ordered) and ordered[t].shape[0] == ordered[start].shape[0]:
                continue
            seg_prey = np.stack(ordered[start:t]) * ARENA   # [T, N, 2] in px
            seg_pred = np.stack(preds[start:t]) * ARENA     # [T, 2] in px
            start = t
            if len(seg_pred) < CLOSING_LAG + 2:
                continue
            prey_v = seg_prey[1:] - seg_prey[:-1]
            pred_v = seg_pred[1:] - seg_pred[:-1]
            values = compute_pursuit_metrics(seg_pred[:-1], seg_prey[:-1],
                                             headings_from_velocity(pred_v[:, 0], pred_v[:, 1]),
                                             headings_from_velocity(prey_v[..., 0], prey_v[..., 1]))
            for key in ("pursuit_alignment", "closing_speed", "reorientation", "nearest_distance"):
                metrics[key].extend(values[key].tolist())

    metrics = {k: np.asarray(v, dtype=float) for k, v in metrics.items()}
    for k, v in metrics.items():
        print(f"    {k:22s} mean {np.nanmean(v):+.4f}  (n={np.isfinite(v).sum()})")
    return metrics


def moving_average(values, window):
    """
    Smooths a time series for plotting only (per-step metrics are very noisy)

    Input: time series, window size
    Output: smoothed series
    """
    values = np.asarray(values, dtype=float)
    if window is None or window <= 1 or len(values) < window:
        return values
    return np.convolve(values, np.ones(window) / window, mode="valid")


def save_figure(fig, save_as):
    """
    Saves a figure as PDF and PNG

    Input: figure, file name without extension
    Output: none
    """
    if save_as is not None:
        fig.savefig(f"{save_as}.pdf", bbox_inches="tight")
        fig.savefig(f"{save_as}.png", dpi=300, bbox_inches="tight")


def plot_metric_boxplot(series, expert_values, ylabel, title, ylim=None, save_as=None):
    """
    Draws the distribution of a per-step metric as one box per model next to the expert
    Same colours as the other plots, white diamond = mean, black line = median

    Input: {model: values}, expert values, axis label, title
    Output: boxplot of the metric
    """

    names = (["Expert"] if expert_values is not None else []) + list(series)
    data = ([expert_values] if expert_values is not None else []) + list(series.values())
    data = [np.asarray(d, dtype=float)[np.isfinite(np.asarray(d, dtype=float))] for d in data]
    box_colors = (["#5E5A5A"] if expert_values is not None else []) + [COLORS[n] for n in series]

    fig, ax = plt.subplots(figsize=(8, 4))
    box = ax.boxplot(data, showfliers=False, showmeans=True, patch_artist=True, widths=0.6,
                     medianprops=dict(color="black"),
                     meanprops=dict(marker="D", markerfacecolor="white", markeredgecolor="black", markersize=5))
    for patch, color in zip(box["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    ax.axhline(0, color="#999999", linewidth=0.6, linestyle=":")
    ax.set_xticks(range(1, len(names) + 1))
    ax.set_xticklabels(names)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)

    plt.tight_layout()
    save_figure(fig, save_as)
    plt.show()

    for name, values in zip(names, data):
        print(f"    {name:8s} mean {np.mean(values):+8.3f}   median {np.median(values):+8.3f}   n={values.size}")



##### 6.1.1 PREDATOR PURSUIT #####

def plot_nearest_prey_distance(gail_metrics=None, bc_metrics=None, couzin_metrics=None, random_metrics=None, expert_metrics=None, save_as=None):
    """
    Plots the predator distance to its nearest prey over time to compare different models

    Input: model metrics, expert metrics
    Output: plot of predator distance to nearest prey over time
    """

    # time dimension
    steps = np.arange(len(gail_metrics[0]))

    # get metrics from simulation
    gail_metrics = gail_metrics[0] if gail_metrics is not None else []
    bc_metrics = bc_metrics[0] if bc_metrics is not None else []
    couzin_metrics = couzin_metrics if couzin_metrics is not None else []
    random_metrics = random_metrics[0] if random_metrics is not None else []
    expert_metrics = expert_metrics if expert_metrics is not None else {}

    fig, ax = plt.subplots(figsize=(8, 4))

    # get nearest prey distance data
    gail_pnd = [m.get("distance_nearest_prey") for m in gail_metrics if "distance_nearest_prey" in m]
    bc_pnd = [m.get("distance_nearest_prey") for m in bc_metrics if "distance_nearest_prey" in m]
    couzin_pnd = [m.get("distance_nearest_prey") for m in couzin_metrics if "distance_nearest_prey" in m]
    random_pnd = [m.get("distance_nearest_prey") for m in random_metrics if "distance_nearest_prey" in m]
    expert_pnd = np.mean(np.asarray(expert_metrics["distance_nearest_prey"], dtype=float) * 2160) if "distance_nearest_prey" in expert_metrics else None
    expert_pnd_std = np.std(np.asarray(expert_metrics["distance_nearest_prey"], dtype=float) * 2160) if "distance_nearest_prey" in expert_metrics else None

    # plot nearest prey distance
    ax.plot(steps, gail_pnd, label="GAIL", color=COLORS["GAIL"], linewidth=1) if len(gail_pnd) > 0 else None
    ax.plot(steps, bc_pnd, label="BC", color=COLORS["BC"], linewidth=1) if len(bc_pnd) > 0 else None
    ax.plot(steps, couzin_pnd, label="Couzin", color=COLORS["Couzin"], linewidth=1) if len(couzin_pnd) > 0 else None
    ax.plot(steps, random_pnd, label="Random", color=COLORS["Random"], linewidth=1) if len(random_pnd) > 0 else None
    ax.axhline(expert_pnd, label="Expert", color="#000000", linewidth=1) if expert_pnd is not None else None
    ax.axhspan(expert_pnd - expert_pnd_std, expert_pnd + expert_pnd_std, color="#5E5A5A", alpha=0.12, linewidth=0) if (expert_pnd is not None and expert_pnd_std is not None) else None
    ax.set_xlabel("Steps")
    ax.set_ylabel("Nearest Prey Distance")
    ax.set_title("Predator Distance to Nearest Prey")
    ax.set_xlim(0, steps[-1])
    ax.set_ylim(0, 2160)
    ax.legend()

    plt.tight_layout()
    save_figure(fig, save_as)
    plt.show()

    return (expert_pnd, expert_pnd_std)


def plot_pursuit_alignment(gail_metrics=None, bc_metrics=None, couzin_metrics=None, random_metrics=None, expert_metrics=None, kind="hist", smooth=10, save_as=None):
    """
    Plots the pursuit alignment to compare different models
    Values are computed with compute_pursuit_alignment

    Input: model metrics, expert metrics, kind ("hist" = distribution, "box" = boxplot,
           "time" = over time), smoothing window
    Output: plot of pursuit alignment
    """

    # compute metric for every model
    models = {"GAIL": gail_metrics, "BC": bc_metrics, "Couzin": couzin_metrics, "Random": random_metrics}
    series = {name: pursuit_metrics_from_simulation(m)["pursuit_alignment"] for name, m in models.items() if m is not None}
    expert_metrics = expert_metrics if expert_metrics is not None else {}

    # get expert data
    expert_values = np.asarray(expert_metrics["pursuit_alignment"], dtype=float) if "pursuit_alignment" in expert_metrics else None
    expert_pa = np.nanmean(expert_values) if expert_values is not None else None
    expert_pa_std = np.nanstd(expert_values) if expert_values is not None else None

    # plot pursuit alignment
    if kind == "box":
        plot_metric_boxplot(series, expert_values, "Pursuit Alignment", "Pursuit Alignment",
                            ylim=(-1, 1), save_as=save_as)

    elif kind == "hist":
        fig, ax = plt.subplots(figsize=(8, 4))
        bins = np.linspace(-1, 1, 21)
        if expert_values is not None:
            ax.hist(expert_values[np.isfinite(expert_values)], bins=bins, density=True,
                    histtype="stepfilled", color="#5E5A5A", alpha=0.30,
                    edgecolor="#000000", linewidth=1.2, label="Expert")
        for name, values in series.items():
            ax.hist(values[np.isfinite(values)], bins=bins, density=True,
                    histtype="step", linewidth=1.5, color=COLORS[name], label=name)
        ax.axvline(0, color="#999999", linewidth=0.6, linestyle=":")
        ax.set_xlabel("Pursuit Alignment")
        ax.set_ylabel("Density")
        ax.set_title("Distribution of Pursuit Alignment")
        ax.set_xlim(-1, 1)
        ax.legend()

        plt.tight_layout()
        save_figure(fig, save_as)
        plt.show()

        # share of steps in which the predator moves towards its nearest prey
        if expert_values is not None:
            v = expert_values[np.isfinite(expert_values)]
            print(f"    {'Expert':8s} mean {np.mean(v):+.3f}   towards nearest prey in {100*np.mean(v > 0):5.1f}% of steps")
        for name, values in series.items():
            v = np.asarray(values, dtype=float)
            v = v[np.isfinite(v)]
            print(f"    {name:8s} mean {np.mean(v):+.3f}   towards nearest prey in {100*np.mean(v > 0):5.1f}% of steps")

    else:
        fig, ax = plt.subplots(figsize=(8, 4))
        for name, values in series.items():
            values = moving_average(values, smooth)
            ax.plot(np.arange(len(values)), values, label=name, color=COLORS[name], linewidth=1)
        ax.axhline(expert_pa, label="Expert", color="#000000", linewidth=1) if expert_pa is not None else None
        ax.axhspan(expert_pa - expert_pa_std, expert_pa + expert_pa_std, color="#5E5A5A", alpha=0.12, linewidth=0) if (expert_pa is not None and expert_pa_std is not None) else None
        ax.set_xlabel("Steps")
        ax.set_ylabel("Pursuit Alignment")
        ax.set_title("Pursuit Alignment Over Time")
        ax.set_ylim(-1, 1)
        ax.legend()

        plt.tight_layout()
        save_figure(fig, save_as)
        plt.show()

    return (expert_pa, expert_pa_std)


def plot_closing_speed(gail_metrics=None, bc_metrics=None, couzin_metrics=None, random_metrics=None, expert_metrics=None, kind="box", smooth=10, save_as=None):
    """
    Plots the closing speed of the predator to compare different models
    Values are computed with Jannik's compute_closing_speed

    Input: model metrics, expert metrics, kind ("box" = distribution, "time" = over time), smoothing window
    Output: boxplot or time plot of closing speed
    """

    # compute metric for every model
    models = {"GAIL": gail_metrics, "BC": bc_metrics, "Couzin": couzin_metrics, "Random": random_metrics}
    series = {name: pursuit_metrics_from_simulation(m)["closing_speed"] for name, m in models.items() if m is not None}
    expert_metrics = expert_metrics if expert_metrics is not None else {}

    # get expert data
    expert_values = np.asarray(expert_metrics["closing_speed"], dtype=float) if "closing_speed" in expert_metrics else None
    expert_cs = np.nanmean(expert_values) if expert_values is not None else None
    expert_cs_std = np.nanstd(expert_values) if expert_values is not None else None

    # plot closing speed
    if kind == "box":
        plot_metric_boxplot(series, expert_values, "Closing Speed [px/step]", "Closing Speed",
                            ylim=(-25, 25), save_as=save_as)   # the predator moves 10 px per step, so +/- 20 is the physical maximum
    else:
        fig, ax = plt.subplots(figsize=(8, 4))
        for name, values in series.items():
            values = moving_average(values, smooth)
            ax.plot(np.arange(len(values)), values, label=name, color=COLORS[name], linewidth=1)
        ax.axhline(expert_cs, label="Expert", color="#000000", linewidth=1) if expert_cs is not None else None
        ax.axhspan(expert_cs - expert_cs_std, expert_cs + expert_cs_std, color="#5E5A5A", alpha=0.12, linewidth=0) if (expert_cs is not None and expert_cs_std is not None) else None
        ax.axhline(0, color="#999999", linewidth=0.6, linestyle=":")
        ax.set_xlabel("Steps")
        ax.set_ylabel("Closing Speed [px/step]")
        ax.set_title("Closing Speed Over Time")
        ax.set_ylim(-25, 25)
        ax.legend()

        plt.tight_layout()
        save_figure(fig, save_as)
        plt.show()

    return (expert_cs, expert_cs_std)


##### 6.1.2 PREY EVASION #####

def plot_distance_to_predator(gail_metrics=None, bc_metrics=None, couzin_metrics=None, random_metrics=None, expert_metrics=None, save_as=None):
    """
    Plots the distance between the prey center and the predator over time to compare different models

    Input: model metrics, expert metrics
    Output: plot of distance to predator over time
    """

    # time dimension
    steps = np.arange(len(gail_metrics[0]))

    # get metrics from simulation
    gail_metrics = gail_metrics[0] if gail_metrics is not None else []
    bc_metrics = bc_metrics[0] if bc_metrics is not None else []
    couzin_metrics = couzin_metrics if couzin_metrics is not None else []
    random_metrics = random_metrics[0] if random_metrics is not None else []
    expert_metrics = expert_metrics if expert_metrics is not None else {}

    fig, ax = plt.subplots(figsize=(8, 4))

    # get distance to predator data
    gail_dtp = [m.get("distance_to_predator") for m in gail_metrics if "distance_to_predator" in m]
    bc_dtp = [m.get("distance_to_predator") for m in bc_metrics if "distance_to_predator" in m]
    couzin_dtp = [m.get("distance_to_predator") for m in couzin_metrics if "distance_to_predator" in m]
    random_dtp = [m.get("distance_to_predator") for m in random_metrics if "distance_to_predator" in m]
    expert_dtp = np.mean(expert_metrics["distance_to_predator"]) * 2160 if "distance_to_predator" in expert_metrics else None
    expert_dtp_std = np.std(expert_metrics["distance_to_predator"]) * 2160 if "distance_to_predator" in expert_metrics else None

    # plot distance to predator
    ax.plot(steps, gail_dtp, label="GAIL", color=COLORS["GAIL"], linewidth=1) if len(gail_dtp) > 0 else None
    ax.plot(steps, bc_dtp, label="BC", color=COLORS["BC"], linewidth=1) if len(bc_dtp) > 0 else None
    ax.plot(steps, couzin_dtp, label="Couzin", color=COLORS["Couzin"], linewidth=1) if len(couzin_dtp) > 0 else None
    ax.plot(steps, random_dtp, label="Random", color=COLORS["Random"], linewidth=1) if len(random_dtp) > 0 else None
    ax.axhline(expert_dtp, label="Expert", color="#000000", linewidth=1) if expert_dtp is not None else None
    ax.axhspan(expert_dtp - expert_dtp_std, expert_dtp + expert_dtp_std, color="#5E5A5A", alpha=0.12, linewidth=0) if (expert_dtp is not None and expert_dtp_std is not None) else None
    ax.set_xlabel("Steps")
    ax.set_ylabel("Distance to Predator")
    ax.set_title("Distance to Predator Over Time")
    ax.set_xlim(0, steps[-1])
    ax.set_ylim(0, 2160)
    ax.legend()

    plt.tight_layout()
    save_figure(fig, save_as)
    plt.show()

    return (expert_dtp, expert_dtp_std)


def plot_escape_alignment(gail_metrics=None, bc_metrics=None, couzin_metrics=None, random_metrics=None, expert_metrics=None, save_as=None):
    """
    Plots the escape alignment of the prey over time to compare different models

    Input: model metrics, expert metrics
    Output: plot of escape alignment over time
    """

    # time dimension
    steps = np.arange(len(gail_metrics[0]))

    # get metrics from simulation
    gail_metrics = gail_metrics[0] if gail_metrics is not None else []
    bc_metrics = bc_metrics[0] if bc_metrics is not None else []
    couzin_metrics = couzin_metrics if couzin_metrics is not None else []
    random_metrics = random_metrics[0] if random_metrics is not None else []
    expert_metrics = expert_metrics if expert_metrics is not None else {}

    fig, ax = plt.subplots(figsize=(8, 4))

    # get escape alignment data
    gail_ea = [m.get("escape_alignment") for m in gail_metrics if "escape_alignment" in m]
    bc_ea = [m.get("escape_alignment") for m in bc_metrics if "escape_alignment" in m]
    couzin_ea = [m.get("escape_alignment") for m in couzin_metrics if "escape_alignment" in m]
    random_ea = [m.get("escape_alignment") for m in random_metrics if "escape_alignment" in m]
    expert_ea = np.mean(expert_metrics["escape_alignment"]) if "escape_alignment" in expert_metrics else None
    expert_ea_std = np.std(expert_metrics["escape_alignment"]) if "escape_alignment" in expert_metrics else None

    # plot escape alignment
    ax.plot(steps, gail_ea, label="GAIL", color=COLORS["GAIL"], linewidth=1) if len(gail_ea) > 0 else None
    ax.plot(steps, bc_ea, label="BC", color=COLORS["BC"], linewidth=1) if len(bc_ea) > 0 else None
    ax.plot(steps, couzin_ea, label="Couzin", color=COLORS["Couzin"], linewidth=1) if len(couzin_ea) > 0 else None
    ax.plot(steps, random_ea, label="Random", color=COLORS["Random"], linewidth=1) if len(random_ea) > 0 else None
    ax.axhline(expert_ea, label="Expert", color="#000000", linewidth=1) if expert_ea is not None else None
    ax.axhspan(expert_ea - expert_ea_std, expert_ea + expert_ea_std, color="#5E5A5A", alpha=0.12, linewidth=0) if (expert_ea is not None and expert_ea_std is not None) else None
    ax.set_xlabel("Steps")
    ax.set_ylabel("Escape Alignment")
    ax.set_title("Escape Alignment Over Time")
    ax.set_xlim(0, steps[-1])
    ax.set_ylim(-1, 1)
    ax.legend()

    plt.tight_layout()
    save_figure(fig, save_as)
    plt.show()

    return (expert_ea, expert_ea_std)


def plot_prey_reorientation(gail_metrics=None, bc_metrics=None, couzin_metrics=None, random_metrics=None, expert_metrics=None, kind="box", smooth=10, save_as=None):
    """
    Plots the prey reorientation R to compare different models
    Values are computed with Jannik's compute_continuous_predator_response_tensor

    Input: model metrics, expert metrics, kind ("box" = distribution, "time" = over time), smoothing window
    Output: boxplot or time plot of prey reorientation
    """

    # compute metric for every model
    models = {"GAIL": gail_metrics, "BC": bc_metrics, "Couzin": couzin_metrics, "Random": random_metrics}
    series = {name: pursuit_metrics_from_simulation(m)["reorientation"] for name, m in models.items() if m is not None}
    expert_metrics = expert_metrics if expert_metrics is not None else {}

    # get expert data
    expert_values = np.asarray(expert_metrics["reorientation"], dtype=float) if "reorientation" in expert_metrics else None
    expert_r = np.nanmean(expert_values) if expert_values is not None else None
    expert_r_std = np.nanstd(expert_values) if expert_values is not None else None

    # plot prey reorientation
    if kind == "box":
        plot_metric_boxplot(series, expert_values, "Prey Reorientation R", "Prey Reorientation",
                            save_as=save_as)
    else:
        fig, ax = plt.subplots(figsize=(8, 4))
        for name, values in series.items():
            values = moving_average(values, smooth)
            ax.plot(np.arange(len(values)), values, label=name, color=COLORS[name], linewidth=1)
        ax.axhline(expert_r, label="Expert", color="#000000", linewidth=1) if expert_r is not None else None
        ax.axhspan(expert_r - expert_r_std, expert_r + expert_r_std, color="#5E5A5A", alpha=0.12, linewidth=0) if (expert_r is not None and expert_r_std is not None) else None
        ax.axhline(0, color="#999999", linewidth=0.6, linestyle=":")
        ax.set_xlabel("Steps")
        ax.set_ylabel("Prey Reorientation R")
        ax.set_title("Prey Reorientation Over Time")
        ax.legend()

        plt.tight_layout()
        save_figure(fig, save_as)
        plt.show()

    return (expert_r, expert_r_std)


#########################
##### SWARM METRICS #####
#########################
# The three panels below are the panels of plot_swarm_metrics, split into one
# function per figure so that every metric can be shown on its own.


##### 6.2.1 POLARIZATION #####

def plot_polarization(gail_metrics=None, bc_metrics=None, couzin_metrics=None, random_metrics=None, expert_metrics=None, save_as=None):
    """
    Plots the polarization of the swarm over time to compare different models

    Input: model metrics, expert metrics
    Output: plot of polarization over time
    """

    # time dimension
    steps = np.arange(len(gail_metrics[0]))

    # get metrics from simulation
    gail_metrics = gail_metrics[0] if gail_metrics is not None else []
    bc_metrics = bc_metrics[0] if bc_metrics is not None else []
    couzin_metrics = couzin_metrics if couzin_metrics is not None else []
    random_metrics = random_metrics[0] if random_metrics is not None else []
    expert_metrics = expert_metrics if expert_metrics is not None else {}

    fig, ax = plt.subplots(figsize=(8, 4))

    # get polarization data
    gail_polarization = [m.get("polarization") for m in gail_metrics if "polarization" in m]
    bc_polarization = [m.get("polarization") for m in bc_metrics if "polarization" in m]
    couzin_polarization = [m.get("polarization") for m in couzin_metrics if "polarization" in m]
    random_polarization = [m.get("polarization") for m in random_metrics if "polarization" in m]
    expert_polarization = np.mean(expert_metrics["polarization"]) if "polarization" in expert_metrics else None
    expert_polarization_std = np.std(expert_metrics["polarization"]) if "polarization" in expert_metrics else None

    # plot polarization
    ax.plot(steps, gail_polarization, label="GAIL", color=COLORS["GAIL"], linewidth=1) if len(gail_polarization) > 0 else None
    ax.plot(steps, bc_polarization, label="BC", color=COLORS["BC"], linewidth=1) if len(bc_polarization) > 0 else None
    ax.plot(steps, couzin_polarization, label="Couzin", color=COLORS["Couzin"], linewidth=1) if len(couzin_polarization) > 0 else None
    ax.plot(steps, random_polarization, label="Random", color=COLORS["Random"], linewidth=1) if len(random_polarization) > 0 else None
    ax.axhline(expert_polarization, label="Expert", color="#000000", linewidth=1) if expert_polarization is not None else None
    ax.axhspan(expert_polarization - expert_polarization_std, expert_polarization + expert_polarization_std, color="#5E5A5A", alpha=0.12, linewidth=0) if (expert_polarization is not None and expert_polarization_std is not None) else None
    ax.set_xlabel("Steps")
    ax.set_ylabel("Polarization")
    ax.set_title("Polarization Over Time")
    ax.set_xlim(0, steps[-1])
    ax.set_ylim(0, 1)
    ax.legend()

    plt.tight_layout()
    save_figure(fig, save_as)
    plt.show()

    return (expert_polarization, expert_polarization_std)


##### 6.2.2 DEGREE OF SPARSITY #####

def plot_degree_of_sparsity(gail_metrics=None, bc_metrics=None, couzin_metrics=None, random_metrics=None, expert_metrics=None, save_as=None):
    """
    Plots the degree of sparsity of the swarm over time to compare different models

    Input: model metrics, expert metrics
    Output: plot of degree of sparsity over time
    """

    # time dimension
    steps = np.arange(len(gail_metrics[0]))

    # get metrics from simulation
    gail_metrics = gail_metrics[0] if gail_metrics is not None else []
    bc_metrics = bc_metrics[0] if bc_metrics is not None else []
    couzin_metrics = couzin_metrics if couzin_metrics is not None else []
    random_metrics = random_metrics[0] if random_metrics is not None else []
    expert_metrics = expert_metrics if expert_metrics is not None else {}

    fig, ax = plt.subplots(figsize=(8, 4))

    # get degree of sparsity data
    gail_dos = [m.get("degree_of_sparsity") for m in gail_metrics if "degree_of_sparsity" in m]
    bc_dos = [m.get("degree_of_sparsity") for m in bc_metrics if "degree_of_sparsity" in m]
    couzin_dos = [m.get("degree_of_sparsity") for m in couzin_metrics if "degree_of_sparsity" in m]
    random_dos = [m.get("degree_of_sparsity") for m in random_metrics if "degree_of_sparsity" in m]
    expert_dos = np.mean(expert_metrics["sparsity"]) * 2160 if "sparsity" in expert_metrics else None
    expert_dos_std = np.std(expert_metrics["sparsity"]) * 2160 if "sparsity" in expert_metrics else None

    # plot degree of sparsity
    ax.plot(steps, gail_dos, label="GAIL", color=COLORS["GAIL"], linewidth=1) if len(gail_dos) > 0 else None
    ax.plot(steps, bc_dos, label="BC", color=COLORS["BC"], linewidth=1) if len(bc_dos) > 0 else None
    ax.plot(steps, couzin_dos, label="Couzin", color=COLORS["Couzin"], linewidth=1) if len(couzin_dos) > 0 else None
    ax.plot(steps, random_dos, label="Random", color=COLORS["Random"], linewidth=1) if len(random_dos) > 0 else None
    ax.axhline(expert_dos, label="Expert", color="#000000", linewidth=1) if expert_dos is not None else None
    ax.axhspan(expert_dos - expert_dos_std, expert_dos + expert_dos_std, color="#5E5A5A", alpha=0.12, linewidth=0) if (expert_dos is not None and expert_dos_std is not None) else None
    ax.set_xlabel("Steps")
    ax.set_ylabel("Degree of Sparsity")
    ax.set_title("Degree of Sparsity Over Time")
    ax.set_xlim(0, steps[-1])
    ax.legend()

    plt.tight_layout()
    save_figure(fig, save_as)
    plt.show()

    return (expert_dos, expert_dos_std)


##### 6.2.3 ANGULAR MOMENTUM #####

def plot_angular_momentum(gail_metrics=None, bc_metrics=None, couzin_metrics=None, random_metrics=None, expert_metrics=None, save_as=None):
    """
    Plots the angular momentum of the swarm over time to compare different models

    Input: model metrics, expert metrics
    Output: plot of angular momentum over time
    """

    # time dimension
    steps = np.arange(len(gail_metrics[0]))

    # get metrics from simulation
    gail_metrics = gail_metrics[0] if gail_metrics is not None else []
    bc_metrics = bc_metrics[0] if bc_metrics is not None else []
    couzin_metrics = couzin_metrics if couzin_metrics is not None else []
    random_metrics = random_metrics[0] if random_metrics is not None else []
    expert_metrics = expert_metrics if expert_metrics is not None else {}

    fig, ax = plt.subplots(figsize=(8, 4))

    # get angular momentum data
    gail_am = [m.get("angular_momentum") for m in gail_metrics if "angular_momentum" in m]
    bc_am = [m.get("angular_momentum") for m in bc_metrics if "angular_momentum" in m]
    couzin_am = [m.get("angular_momentum") for m in couzin_metrics if "angular_momentum" in m]
    random_am = [m.get("angular_momentum") for m in random_metrics if "angular_momentum" in m]
    expert_am = np.mean(expert_metrics["angular_momentum"]) * 2160 if "angular_momentum" in expert_metrics else None
    expert_am_std = np.std(expert_metrics["angular_momentum"]) * 2160 if "angular_momentum" in expert_metrics else None

    # plot angular momentum
    ax.plot(steps, gail_am, label="GAIL", color=COLORS["GAIL"], linewidth=1) if len(gail_am) > 0 else None
    ax.plot(steps, bc_am, label="BC", color=COLORS["BC"], linewidth=1) if len(bc_am) > 0 else None
    ax.plot(steps, couzin_am, label="Couzin", color=COLORS["Couzin"], linewidth=1) if len(couzin_am) > 0 else None
    ax.plot(steps, random_am, label="Random", color=COLORS["Random"], linewidth=1) if len(random_am) > 0 else None
    ax.axhline(expert_am, label="Expert", color="#000000", linewidth=1) if expert_am is not None else None
    ax.axhspan(expert_am - expert_am_std, expert_am + expert_am_std, color="#5E5A5A", alpha=0.12, linewidth=0) if (expert_am is not None and expert_am_std is not None) else None
    ax.set_xlabel("Steps")
    ax.set_ylabel("Angular Momentum")
    ax.set_title("Angular Momentum Over Time")
    ax.set_xlim(0, steps[-1])
    ax.legend()

    plt.tight_layout()
    save_figure(fig, save_as)
    plt.show()

    return (expert_am, expert_am_std)


##########################
##### ATTENTION MAPS #####
##########################

def compute_pin_an_maps(pin, an, grid_size=100, n_orient=100, role="prey_pred", v_dir=None, v_scale=1.0, max_turn=0.314):
    """
    Computes policy and attention maps for PIN and AN models

    Input: PIN and AN, grid size, number of orientations, role,
           v_dir (None = average over all neighbour directions, else one direction in degrees),
           v_scale (neighbour speed, 1.0 = maximum, ~0.15 = typical in the expert data),
           max_turn
           
    Output: xs, ys, action map, attention map
    """

    pin.to("cpu").eval(); an.to("cpu").eval()

    # relative positions grid
    xs = np.linspace(-1, 1, grid_size)
    ys = np.linspace(-1, 1, grid_size)

    # sample relative velocity directions (all of them, or one fixed direction)
    if v_dir is None:
        thetas = torch.linspace(-np.pi, np.pi, n_orient + 1)[:-1]
    else:
        thetas = torch.tensor([np.radians(v_dir)], dtype=torch.float32)

    n_dirs = len(thetas)
    rel_vx = (v_scale * torch.cos(thetas)).unsqueeze(1)
    rel_vy = (v_scale * torch.sin(thetas)).unsqueeze(1)
    active = torch.ones((n_dirs, 1))

    # initialize maps
    action_map = np.zeros((grid_size, grid_size), dtype=np.float32)
    attn_map   = np.zeros((grid_size, grid_size), dtype=np.float32)

    for ix, x in enumerate(xs):
        for iy, y in enumerate(ys):
            # build input tensor for this grid cell
            dx = torch.full((n_dirs, 1), float(x))
            dy = torch.full((n_dirs, 1), float(y))

            if role == "predator": # 5 features
                inputs = torch.cat([dx, dy, rel_vx, rel_vy, active], dim=1)
            else: # 6 features
                # to handle flag feature for prey
                flag = torch.ones((n_dirs, 1)) if role == "prey_pred" else torch.zeros((n_dirs, 1))
                inputs = torch.cat([flag, dx, dy, rel_vx, rel_vy, active], dim=1)

            with torch.no_grad():
                mu, sigma = pin(inputs)
                # convert to the actual turn the environment would apply, in degrees
                turn = (torch.sigmoid(mu.reshape(-1)) - 0.5) * 2 * max_turn * 180 / np.pi
                action_map[iy, ix] = turn.mean().item()
                attn_map[iy, ix]   = an(inputs).reshape(-1).mean().item()

    return xs, ys, action_map, minmax_norm(attn_map)


def plot_policy_maps(xs, ys, action_map, attn_map, role="predator", img_path=None, titles=None):
    """
    Plots policy and attention maps for PIN and AN models

    Input: xs, ys, action map, attention map, role, image path
    Output: plots of policy and attention maps
    """

    # color palettes
    cmap_pin_color = "RdBu_r"
    cmap_an_color = "magma"

    # create meshgrid for plotting
    x, y = np.meshgrid(xs, ys)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # change titles
    t0, t1 = titles if titles else (f"[{role.upper()}] Pairwise-Interaction Map",
                                    f"[{role.upper()}] Attention Map")

    # action map in degrees
    scaled_action_map = action_map
    vmax_act = np.nanmax(np.abs(scaled_action_map)) # symmetric color scale
    norm_act = colors.TwoSlopeNorm(vcenter=0, vmin=-vmax_act, vmax=vmax_act) # centered at 0

    # plot PIN map
    im0 = axes[0].contourf(x, y, scaled_action_map, levels=30, cmap=cmap_pin_color, norm=norm_act)
    axes[0].set_title(t0)
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    plt.colorbar(im0, ax=axes[0], label="turn (degrees)")

    # plot AN map
    im1 = axes[1].contourf(x, y, attn_map, levels=30, cmap=cmap_an_color, vmin=0, vmax=1)
    axes[1].set_title(t1)
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    plt.colorbar(im1, ax=axes[1], label="attention")

    # set same axis ranges
    x_range = (xs.min(), xs.max())
    y_range = (ys.min(), ys.max())
    for ax in axes:
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)

    # add agent icon in center (alignment of picture and action map is correct)
    if img_path is not None and os.path.exists(img_path):
        icon = mpimg.imread(img_path)
        target_px = 40                  
        zoom = target_px / max(icon.shape[:2])  
        imgbox = OffsetImage(icon, zoom=zoom)
        center = (0, 0)
        for ax in axes:
            ab = AnnotationBbox(imgbox, center, frameon=False, xycoords='data')
            ax.add_artist(ab)

    plt.tight_layout()
    plt.show()


def plot_trajectory(metrics, role="Pred & Prey", title=None):
    """
    Plots the trajectories of predator and prey agents

    Input: metrics, role
    Output: plots of trajectories
    """

    metrics = metrics[0]

    # extract pred positions
    xs_pred = np.array([m["xs"][0] for m in metrics])
    ys_pred = np.array([m["ys"][0] for m in metrics])

    # extract prey positions
    xs_prey = np.stack([m["xs"][1:] for m in metrics])
    ys_prey = np.stack([m["ys"][1:] for m in metrics])

    plt.figure(figsize=(6,6))

    # plot trajectories of predator
    if role in ('predator', 'Pred & Prey'):
        plt.plot(xs_pred, ys_pred, color='red', label='Predator')

    # plot trajectories of prey
    if role in ('prey', 'Pred & Prey'):
        _, n_agents = xs_prey.shape
        for i in range(n_agents):
            plt.plot(xs_prey[:, i], ys_prey[:, i], color='gray', alpha=0.6)

    plt.title(title or f'{role} - Trajectory Plot')
    plt.tight_layout()
    plt.show()


