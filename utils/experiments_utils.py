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
import numpy as np
import pandas as pd
from utils.sim_utils import *
from utils.eval_utils import *
from utils.train_utils import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.colors as colors
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

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


def plot_swarm_metrics(gail_metrics=None, bc_metrics=None, couzin_metrics=None, random_metrics=None, expert_metrics=None):
    """
    Plots swarm metrics over time to compare different models
    Code allows dynamic inclusion/exclusion of different models

    Input: model metrics
    Output: plots of polarization, angular momentum, degree of sparsity over time
    """

    # time dimension
    steps = np.arange(len(gail_metrics[0]))
    fig, axes = plt.subplots(1, 3, figsize=(24, 5))

    # get metrics from simulation
    gail_metrics = gail_metrics[0] if gail_metrics is not None else []
    bc_metrics = bc_metrics[0] if bc_metrics is not None else []
    couzin_metrics = couzin_metrics if couzin_metrics is not None else []
    random_metrics = random_metrics[0] if random_metrics is not None else []
    expert_metrics = expert_metrics if expert_metrics is not None else {}

    # get polarization data
    gail_polarization = [m.get("polarization") for m in gail_metrics if "polarization" in m]
    bc_polarization   = [m.get("polarization") for m in bc_metrics   if "polarization" in m]
    couzin_polarization = [m.get("polarization") for m in couzin_metrics if "polarization" in m]
    random_polarization = [m.get("polarization") for m in random_metrics if "polarization" in m]
    expert_polarization = np.mean(expert_metrics["polarization"]) if "polarization" in expert_metrics else None
    expert_polarization_std = np.std(expert_metrics["polarization"]) if "polarization" in expert_metrics else None

    # plot polarization
    axes[0].plot(steps, gail_polarization, label="GAIL", color="#8B0000", linewidth=1) if len(gail_polarization) > 0 else None
    axes[0].plot(steps, bc_polarization, label="BC", color="#F08080", linewidth=1) if len(bc_polarization) > 0 else None
    axes[0].plot(steps, couzin_polarization, label="Couzin", color="#003366", linewidth=1) if len(couzin_polarization) > 0 else None
    axes[0].plot(steps, random_polarization, label="Random", color="#7EC8E3", linewidth=1) if len(random_polarization) > 0 else None
    axes[0].axhline(expert_polarization, label="Expert", color="#000000", linewidth=1) if expert_polarization is not None else None
    axes[0].axhspan(expert_polarization - expert_polarization_std, expert_polarization + expert_polarization_std, color="#5E5A5A", alpha=0.12, linewidth=0) if (expert_polarization is not None and expert_polarization_std is not None) else None
    axes[0].set_xlabel("Steps", fontsize=14)
    axes[0].set_ylabel("Polarization", fontsize=14)
    axes[0].set_title("Polarization Over Time", fontsize=18)
    axes[0].set_xlim(0, steps[-1])
    axes[0].set_ylim(0, 1)
    axes[0].legend()

    # get angular momentum data
    gail_am = [m.get("angular_momentum") for m in gail_metrics if "angular_momentum" in m] if len(gail_metrics)>0 else []
    bc_am = [m.get("angular_momentum") for m in bc_metrics if "angular_momentum" in m] if len(bc_metrics)>0 else []
    couzin_am = [m.get("angular_momentum") for m in couzin_metrics if "angular_momentum" in m] if len(couzin_metrics)>0 else []
    random_am = [m.get("angular_momentum") for m in random_metrics if "angular_momentum" in m] if len(random_metrics)>0 else []
    expert_am = np.mean(expert_metrics["angular_momentum"]) * 2160 if "angular_momentum" in expert_metrics else None
    expert_am_std = np.std(expert_metrics["angular_momentum"]) * 2160 if "angular_momentum" in expert_metrics else None

    # plot angular momentum
    axes[1].plot(steps, gail_am, label="GAIL", color="#8B0000", linewidth=1) if (gail_am is not None and len(gail_am) > 0) else None
    axes[1].plot(steps, bc_am, label="BC", color="#F08080", linewidth=1) if (bc_am is not None and len(bc_am) > 0) else None
    axes[1].plot(steps, couzin_am, label="Couzin", color="#003366", linewidth=1) if (couzin_am is not None and len(couzin_am) > 0) else None
    axes[1].plot(steps, random_am, label="Random", color="#7EC8E3", linewidth=1) if (random_am is not None and len(random_am) > 0) else None
    axes[1].axhline(expert_am, label="Expert", color="#000000", linewidth=1) if expert_am is not None else None
    axes[1].axhspan(expert_am - expert_am_std, expert_am + expert_am_std, color="#5E5A5A", alpha=0.12, linewidth=0) if (expert_am is not None and expert_am_std is not None) else None
    axes[1].set_xlabel("Steps", fontsize=14)
    axes[1].set_ylabel("Angular Momentum", fontsize=14)
    axes[1].set_title("Angular Momentum Over Time", fontsize=18)
    axes[1].set_xlim(0, steps[-1])
    axes[1].legend()

    # get degree of sparsity data
    gail_dos = [m.get("degree_of_sparsity") for m in gail_metrics if "degree_of_sparsity" in m] if len(gail_metrics)>0 else []
    bc_dos = [m.get("degree_of_sparsity") for m in bc_metrics if "degree_of_sparsity" in m] if len(bc_metrics)>0 else []
    couzin_dos = [m.get("degree_of_sparsity") for m in couzin_metrics if "degree_of_sparsity" in m] if len(couzin_metrics)>0 else []
    random_dos = [m.get("degree_of_sparsity") for m in random_metrics if "degree_of_sparsity" in m] if len(random_metrics)>0 else []
    expert_dos = np.mean(expert_metrics["sparsity"]) * 2160 if "sparsity" in expert_metrics else None
    expert_dos_std = np.std(expert_metrics["sparsity"]) * 2160 if "sparsity" in expert_metrics else None

    # plot degree of sparsity
    axes[2].plot(steps, gail_dos, label="GAIL", color="#8B0000", linewidth=1) if len(gail_dos) > 0 else None
    axes[2].plot(steps, bc_dos, label="BC", color="#F08080", linewidth=1) if len(bc_dos) > 0 else None
    axes[2].plot(steps, couzin_dos, label="Couzin", color="#003366", linewidth=1) if len(couzin_dos) > 0 else None 
    axes[2].plot(steps, random_dos, label="Random", color="#7EC8E3", linewidth=1) if len(random_dos) > 0 else None 
    axes[2].axhline(expert_dos, label="Expert", color="#000000", linewidth=1) if expert_dos is not None else None 
    axes[2].axhspan(expert_dos - expert_dos_std, expert_dos + expert_dos_std, color="#5E5A5A", alpha=0.12, linewidth=0) if (expert_dos is not None and expert_dos_std is not None) else None
    axes[2].set_xlabel("Steps", fontsize=14)
    axes[2].set_ylabel("Degree of Sparsity", fontsize=14)
    axes[2].set_title("Degree of Sparsity Over Time", fontsize=18)
    axes[2].set_xlim(0, steps[-1])
    axes[2].legend()

    plt.tight_layout()
    plt.show()

    return (expert_polarization, expert_polarization_std, expert_am, expert_am_std, expert_dos, expert_dos_std)


def plot_pred_prey_metrics(gail_metrics=None, bc_metrics=None, couzin_metrics=None, random_metrics=None, expert_metrics=None):
    """
    Plots predator-related metrics over time to compare different models
    Code allows dynamic inclusion/exclusion of different models

    Input: model metrics
    Output: plots of distance to predator and escape alignment over time
    """

    # time dimension
    steps = np.arange(len(gail_metrics[0]))

    # get metrics from simulation
    gail_metrics = gail_metrics[0] if gail_metrics is not None else []
    bc_metrics = bc_metrics[0] if bc_metrics is not None else []
    couzin_metrics = couzin_metrics if couzin_metrics is not None else []
    random_metrics = random_metrics[0] if random_metrics is not None else []
    expert_metrics = expert_metrics if expert_metrics is not None else {}

    fig, axes = plt.subplots(1, 3, figsize=(18, 4))

    # get distance to predator data
    gail_dtp = [m.get("distance_to_predator") for m in gail_metrics if "distance_to_predator" in m] 
    bc_dtp = [m.get("distance_to_predator") for m in bc_metrics if "distance_to_predator" in m]
    random_dtp = [m.get("distance_to_predator") for m in random_metrics if "distance_to_predator" in m]
    couzin_dtp = [m.get("distance_to_predator") for m in couzin_metrics if "distance_to_predator" in m]
    expert_dtp = np.mean(expert_metrics["distance_to_predator"]) * 2160 if "distance_to_predator" in expert_metrics else None
    expert_dtp_std = np.std(expert_metrics["distance_to_predator"]) * 2160 if "distance_to_predator" in expert_metrics else None

    # plot distance to predator
    axes[0].plot(steps, gail_dtp, label="GAIL", color="#8B0000", linewidth=1) if len(gail_dtp) > 0 else None
    axes[0].plot(steps, bc_dtp, label="BC", color="#F08080", linewidth=1) if len(bc_dtp) > 0 else None
    axes[0].plot(steps, couzin_dtp, label="Couzin", color="#003366", linewidth=1) if len(couzin_dtp) > 0 else None
    axes[0].plot(steps, random_dtp, label="Random", color="#7EC8E3", linewidth=1) if len(random_dtp) > 0 else None
    axes[0].axhline(expert_dtp, label="Expert", color="#000000", linewidth=1) if expert_dtp is not None else None
    axes[0].axhspan(expert_dtp - expert_dtp_std, expert_dtp + expert_dtp_std, color="#5E5A5A", alpha=0.12, linewidth=0) if (expert_dtp is not None and expert_dtp_std is not None) else None
    axes[0].set_xlabel("Steps")
    axes[0].set_ylabel("Distance to Predator")
    axes[0].set_title("Distance to Predator Over Time")
    axes[0].set_ylim(0, 2160)
    axes[0].legend()


    # get nearest prey distance data
    gail_pnd = [m.get("distance_nearest_prey") for m in gail_metrics if "distance_nearest_prey" in m]
    bc_pnd = [m.get("distance_nearest_prey") for m in bc_metrics   if "distance_nearest_prey" in m]
    couzin_pnd = [m.get("distance_nearest_prey") for m in couzin_metrics if "distance_nearest_prey" in m]
    random_pnd = [m.get("distance_nearest_prey") for m in random_metrics if "distance_nearest_prey" in m]
    expert_pnd = np.mean(np.asarray(expert_metrics["distance_nearest_prey"], dtype=float) * 2160) if "distance_nearest_prey" in expert_metrics else None
    expert_pnd_std = np.std(np.asarray(expert_metrics["distance_nearest_prey"], dtype=float) * 2160) if "distance_nearest_prey" in expert_metrics else None

    # plot nearest prey distance
    axes[1].plot(steps, gail_pnd, label="GAIL", color="#8B0000", linewidth=1) if len(gail_pnd) > 0 else None
    axes[1].plot(steps, bc_pnd, label="BC", color="#F08080", linewidth=1) if len(bc_pnd) > 0 else None
    axes[1].plot(steps, couzin_pnd, label="Couzin", color="#003366", linewidth=1) if len(couzin_pnd) > 0 else None
    axes[1].plot(steps, random_pnd, label="Random", color="#7EC8E3", linewidth=1) if len(random_pnd) > 0 else None
    axes[1].axhline(expert_pnd, label="Expert", color="#000000", linewidth=1) if expert_pnd is not None else None
    axes[1].axhspan(expert_pnd - expert_pnd_std, expert_pnd + expert_pnd_std, color="#5E5A5A", alpha=0.12, linewidth=0) if (expert_pnd is not None and expert_pnd_std is not None) else None
    axes[1].set_xlabel("Steps")
    axes[1].set_ylabel("Nearest Prey Distance")
    axes[1].set_title("Predator Distance to Nearest Prey")
    axes[1].set_xlim(0, steps[-1])
    axes[1].set_ylim(0, 2160)
    axes[1].legend()


    # get escape alignment data
    gail_ea = [m.get("escape_alignment") for m in gail_metrics if "escape_alignment" in m]
    bc_ea = [m.get("escape_alignment") for m in bc_metrics if "escape_alignment" in m]
    couzin_ea = [m.get("escape_alignment") for m in couzin_metrics if "escape_alignment" in m]
    random_ea = [m.get("escape_alignment") for m in random_metrics if "escape_alignment" in m]
    expert_ea = np.mean(expert_metrics["escape_alignment"]) if "escape_alignment" in expert_metrics else None
    expert_ea_std = np.std(expert_metrics["escape_alignment"]) if "escape_alignment" in expert_metrics else None

    # plot escape alignment
    axes[2].plot(steps, gail_ea, label="GAIL", color="#8B0000", linewidth=1) if len(gail_ea) > 0 else None
    axes[2].plot(steps, bc_ea, label="BC", color="#F08080", linewidth=1) if len(bc_ea) > 0 else None
    axes[2].plot(steps, couzin_ea, label="Couzin", color="#003366", linewidth=1) if len(couzin_ea) > 0 else None
    axes[2].plot(steps, random_ea, label="Random", color="#7EC8E3", linewidth=1) if len(random_ea) > 0 else None
    axes[2].axhline(expert_ea, label="Expert", color="#000000", linewidth=1) if expert_ea is not None else None
    axes[2].axhspan(expert_ea - expert_ea_std, expert_ea + expert_ea_std, color="#5E5A5A", alpha=0.12, linewidth=0) if (expert_ea is not None and expert_ea_std is not None) else None
    axes[2].set_xlabel("Steps")
    axes[2].set_ylabel("Escape Alignment")
    axes[2].set_title("Escape Alignment Over Time")
    axes[2].set_xlim(0, steps[-1])
    axes[2].set_ylim(-1, 1)
    axes[2].legend()

    plt.tight_layout()
    plt.show()

    return (expert_dtp, expert_dtp_std, expert_pnd, expert_pnd_std, expert_ea, expert_ea_std)



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


