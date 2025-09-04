#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import copy
import time
import csv
from datetime import datetime
import pickle
import numpy as np
from tqdm import tqdm

import torch
from tensorboardX import SummaryWriter

from options import args_parser
from update import LocalUpdate, test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar
from utils import get_dataset


# ---------------------------
# Small utilities
# ---------------------------
def parse_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    return s in ("1", "true", "yes", "y", "on")


# ---------------------------
# Weighted aggregation helper
# ---------------------------
def weighted_average_weights(local_weights, client_ids, weights_map):
    avg_w = copy.deepcopy(local_weights[0])
    for k in avg_w.keys():
        avg_w[k].zero_()
    for sd, cid in zip(local_weights, client_ids):
        w = float(weights_map[cid])
        for k in avg_w.keys():
            avg_w[k] += w * sd[k]
    return avg_w


# ---------------------------
# Fairness signal (no Jain): accuracy gap
# g_t ∈ [0,1], higher => less fair
# ---------------------------
def fairness_signal_gap(acc_vec: np.ndarray, gap_norm: float = 0.15):
    acc = np.asarray(acc_vec, dtype=float)
    if acc.size == 0:
        return 0.0, 0.0
    gap = float(np.mean(acc) - np.min(acc))
    g = gap / max(1e-12, gap_norm)
    return float(np.clip(g, 0.0, 1.0)), float(gap)


# ---------------------------
# Adaptive alpha (modular)
# ---------------------------
_ALPHA_GAIN   = 1.0
_ALPHA_GAMMA  = 1.0
_ALPHA_BETA   = 0.6
_ALPHA_WARMUP = 1

def _shape_signal(g: float) -> float:
    g = float(np.clip(g, 0.0, 1.0))
    g_pow = g ** _ALPHA_GAMMA
    g_amp = _ALPHA_GAIN * g_pow
    return float(np.clip(g_amp, 0.0, 1.0))

def compute_adaptive_alpha(prev_alpha: float,
                           acc_vec: np.ndarray,
                           epoch: int,
                           alpha_min: float,
                           alpha_max: float):
    g_t, gap_raw = fairness_signal_gap(acc_vec, gap_norm=0.15)
    g_shaped = _shape_signal(g_t)
    alpha_raw = alpha_min + (alpha_max - alpha_min) * g_shaped
    if epoch < _ALPHA_WARMUP:
        alpha_t = alpha_raw
    else:
        alpha_t = (1.0 - _ALPHA_BETA) * float(prev_alpha) + _ALPHA_BETA * float(alpha_raw)
    alpha_t = float(np.clip(alpha_t, alpha_min, alpha_max))
    return alpha_t, float(alpha_raw), float(g_t), float(g_shaped), float(gap_raw)


# ---------------------------
# Client selection & weights
# ---------------------------
def select_clients(Q: np.ndarray, N: int, m: int, fcfl_enabled: bool, fcfl_r: float, cold_start: bool):
    if (not fcfl_enabled) or cold_start:
        idxs_users = np.sort(np.random.choice(range(N), m, replace=False))
        sel_by_q = np.array([], dtype=int)
        sel_rand = idxs_users.copy()
        return idxs_users, sel_by_q, sel_rand

    # Use floor and force k_q < m for m>1 (avoid all-Q)
    k_q = int(np.floor(float(fcfl_r) * m + 1e-12))
    if m > 1:
        k_q = min(k_q, m - 1)
    else:
        k_q = 0
    k_q = max(0, k_q)
    k_rand = m - k_q

    all_ids = np.arange(N, dtype=int)
    top_order = np.argsort(-Q)  # descending Q
    top_part = top_order[:k_q].astype(int)

    remaining = np.setdiff1d(all_ids, top_part, assume_unique=False)
    rand_part = np.random.choice(remaining, size=k_rand, replace=False) if k_rand > 0 else np.array([], dtype=int)

    idxs_users = np.sort(np.concatenate([top_part, rand_part]).astype(int))
    sel_by_q = np.sort(top_part.astype(int))
    sel_rand = np.sort(rand_part.astype(int))
    return idxs_users, sel_by_q, sel_rand

def compute_round_weights(Q: np.ndarray, idxs_users: np.ndarray, n_i: np.ndarray) -> np.ndarray:
    if np.allclose(Q[idxs_users], 0):
        denom = n_i[idxs_users].sum()
        return n_i[idxs_users] / (denom if denom > 0 else 1.0)
    denom = Q[idxs_users].sum()
    return Q[idxs_users] / (denom if denom > 0 else 1.0)


# ======================================================================
# Main
# ======================================================================
if __name__ == '__main__':
    # Reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    start_time = time.time()
    os.makedirs('../logs', exist_ok=True)
    logger = SummaryWriter('../logs')

    # ---------------------------
    # Parse args & device
    # ---------------------------
    args = args_parser()

    # Minimal FCFL flags (+ new print flag)
    if not hasattr(args, 'fcfl'):
        args.fcfl = 1
    if not hasattr(args, 'fcfl_r'):
        args.fcfl_r = 0.7
    if not hasattr(args, 'fcfl_adaptive_alpha'):
        args.fcfl_adaptive_alpha = True
    if not hasattr(args, 'fcfl_alpha_min'):
        args.fcfl_alpha_min = 0.10
    if not hasattr(args, 'fcfl_alpha_max'):
        args.fcfl_alpha_max = 0.80
    if not hasattr(args, 'fcfl_print_selection'):
        args.fcfl_print_selection = False

    # NEW: adaptive r defaults
    if not hasattr(args, 'fcfl_adaptive_r'):
        args.fcfl_adaptive_r = False
    if not hasattr(args, 'fcfl_r_min'):
        args.fcfl_r_min = 0.20
    if not hasattr(args, 'fcfl_r_max'):
        args.fcfl_r_max = 0.95

    # Parse/clip
    args.fcfl_r = float(np.clip(args.fcfl_r, 0.0, 1.0))
    args.fcfl_adaptive_alpha = parse_bool(args.fcfl_adaptive_alpha)
    args.fcfl_print_selection = parse_bool(args.fcfl_print_selection)
    args.fcfl_adaptive_r = parse_bool(args.fcfl_adaptive_r)

    args.fcfl_r_min = float(np.clip(args.fcfl_r_min, 0.0, 1.0))
    args.fcfl_r_max = float(np.clip(args.fcfl_r_max, 0.0, 1.0))
    if args.fcfl_r_min > args.fcfl_r_max:
        args.fcfl_r_min, args.fcfl_r_max = args.fcfl_r_max, args.fcfl_r_min

    args.fcfl_alpha_min = float(max(0.0, args.fcfl_alpha_min))
    args.fcfl_alpha_max = float(min(1.0, max(args.fcfl_alpha_min, args.fcfl_alpha_max)))

    # Device
    if args.gpu is not None and torch.cuda.is_available():
        torch.cuda.set_device(int(args.gpu))
    else:
        args.gpu = None
    device = torch.device(f"cuda:{0 if args.gpu is None else args.gpu}" if torch.cuda.is_available() else "cpu")
    args.device = device

    # ---------------------------
    # Data & model
    # ---------------------------
    train_dataset, test_dataset, user_groups = get_dataset(args)

    if args.model == 'cnn':
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            global_model = CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar':
            global_model = CNNCifar(args=args)
        else:
            raise ValueError("Unsupported dataset for cnn")
    elif args.model == 'mlp':
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
        global_model = MLP(dim_in=len_in, dim_hidden=64, dim_out=args.num_classes)
    else:
        raise ValueError("Unrecognized model")

    global_model.to(device)
    global_model.train()

    # ---------------------------
    # FCFL server state
    # ---------------------------
    N = args.num_users
    m = max(int(round(args.frac * N)), 1)
    n_i = np.array([len(user_groups[i]) for i in range(N)], dtype=float)

    Q      = np.zeros(N, dtype=float)
    omega  = np.zeros(N, dtype=float)
    x_prev = np.zeros(N, dtype=int)
    Acc_hat_t = 0.0

    # alpha initialization
    alpha_t = 0.5 * (args.fcfl_alpha_min + args.fcfl_alpha_max)

    # r state (EMA)
    r_t = float(args.fcfl_r)
    R_EMA = 0.10  # smoothing for adaptive r

    # ---------------------------
    # Run CSV logger
    # ---------------------------
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_name = f"log_{ts}_dataset_{args.dataset}_model_{args.model}_users[{args.num_users}]_frac[{args.frac}]_fcfl[{int(bool(args.fcfl))}]_aa[{int(bool(args.fcfl_adaptive_alpha))}]_alpha[{getattr(args,'fcfl_alpha',alpha_t)}]_alpha_min[{args.fcfl_alpha_min}]_alpha_max[{args.fcfl_alpha_max}]_r[{args.fcfl_r}]_m[{m}]_E[{args.epochs}]_Le[{args.local_ep}]_B[{args.local_bs}]_lr[{args.lr}]_mom[{args.momentum}]_opt[{args.optimizer}]_iid[{args.iid}]_unequal[{args.unequal}]_seed[{args.seed}]_cpr[{args.clients_per_round}]_ar[{args.fcfl_adaptive_r}]_training.csv"
    print(f"Logging to {log_name}")
    log_path = os.path.join("../logs", log_name)

    print(f"Starting training: fcfl={args.fcfl}, aa={args.fcfl_adaptive_alpha}, "
          f"alpha_min={args.fcfl_alpha_min}, alpha_max={args.fcfl_alpha_max}, r={args.fcfl_r}")

    with open(log_path, "w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow([
            "epoch", "mean_acc", "best10_mean", "worst10_mean",
            "unfairness_gap", "unfairness_signal",
            "alpha_t", "alpha_raw", "r",
            "acc_variance",
            "q_mean", "q_std", "q_min", "q_max",
            "sel_total", "sel_by_q", "sel_random",
            "acc_hat_t"
        ])

        # ---------------------------
        # Training rounds
        # ---------------------------
        for epoch in range(args.epochs):
            # (A) Pre-round evaluation
            Acc_t = np.zeros(N, dtype=float)
            for i in range(N):
                lu = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[i], logger=None)
                acc_i, _ = lu.inference(model=global_model)
                Acc_t[i] = acc_i

            mean_acc = float(np.mean(Acc_t))
            var_acc  = float(np.var(Acc_t))
            k = max(1, int(0.1 * N))
            worst10 = float(np.mean(np.partition(Acc_t, k-1)[:k]))
            best10  = float(np.mean(np.partition(Acc_t, -k)[-k:]))

            # Always compute fairness signal for both alpha and r logic
            g_norm, gap_raw = fairness_signal_gap(Acc_t, gap_norm=0.15)
            g_shaped = _shape_signal(g_norm)

            # (B) FCFL policy branch
            if bool(args.fcfl):
                # --- alpha branch ---
                if args.fcfl_adaptive_alpha:
                    alpha_t, alpha_raw, g_norm, g_shaped, gap_raw = compute_adaptive_alpha(
                        prev_alpha=alpha_t, acc_vec=Acc_t, epoch=epoch,
                        alpha_min=args.fcfl_alpha_min, alpha_max=args.fcfl_alpha_max
                    )
                else:
                    # freeze alpha at provided value (clipped)
                    alpha_const = float(np.clip(getattr(args, "fcfl_alpha", alpha_t),
                                                args.fcfl_alpha_min, args.fcfl_alpha_max))
                    alpha_t = alpha_const
                    alpha_raw = alpha_const  # for logging

                # Update queues (skip cold start)
                if epoch > 0:
                    u_ft = np.maximum(Acc_hat_t - Acc_t, 0.0)
                    Q = np.maximum(Q + alpha_t * u_ft - omega * x_prev, 0.0)

                # --- r branch ---
                if args.fcfl_adaptive_r:
                    r_target = args.fcfl_r_min + (args.fcfl_r_max - args.fcfl_r_min) * float(g_shaped)
                    r_target = float(np.clip(r_target, args.fcfl_r_min, args.fcfl_r_max))
                    r_t = (1.0 - R_EMA) * r_t + R_EMA * r_target
                else:
                    r_t = float(args.fcfl_r)

                # Effective r used by selection (never allow all-Q when m>1)
                r_cap_eff = (m - 1) / max(1.0, m) if m > 1 else args.fcfl_r_max
                r_eff = float(np.clip(r_t, args.fcfl_r_min, min(args.fcfl_r_max, r_cap_eff)))
                logged_r = r_eff

                cold_start = (epoch == 0 and np.allclose(Q, 0))
                idxs_users, selected_by_Q, selected_random = select_clients(
                    Q=Q, N=N, m=m, fcfl_enabled=True, fcfl_r=r_eff, cold_start=cold_start
                )
            else:
                # Vanilla FedAvg
                idxs_users = np.sort(np.random.choice(range(N), m, replace=False))
                selected_by_Q = np.array([], dtype=int)
                selected_random = idxs_users.copy()
                alpha_raw = alpha_t  # not used, but logged
                logged_r = 0.0

            # Optional per-epoch prints (only if flag set)
            if args.fcfl_print_selection:
                q_pairs_sorted = sorted([(int(i), float(Q[i])) for i in range(N)],
                                        key=lambda x: x[1], reverse=True)
                print(f"[Round {epoch+1:03d}] Q (desc): {q_pairs_sorted}")
                print(f"[Round {epoch+1:03d}] selected_by_Q({len(selected_by_Q)}): {selected_by_Q.tolist()}")
                print(f"[Round {epoch+1:03d}] selected_random({len(selected_random)}): {selected_random.tolist()}")
                print(f"[Round {epoch+1:03d}] selected_all({len(idxs_users)}): {idxs_users.tolist()}")

            # (C) Aggregation weights
            w_sel = compute_round_weights(Q=Q, idxs_users=idxs_users, n_i=n_i)

            # Prepare next-round penalty terms
            omega = np.zeros(N, dtype=float)
            omega[idxs_users] = w_sel
            x_prev = np.zeros(N, dtype=int)
            x_prev[idxs_users] = 1

            # (D) Local training
            local_weights, client_ids, local_train_acc, local_losses = [], [], [], []
            for cid in idxs_users:
                lu = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[cid], logger=None)
                w_i, loss_i = lu.update_weights(model=copy.deepcopy(global_model), global_round=epoch)
                local_losses.append(loss_i)

                tmp = copy.deepcopy(global_model)
                tmp.load_state_dict(w_i)
                acc_i, _ = lu.inference(model=tmp)

                local_weights.append(w_i)
                client_ids.append(cid)
                local_train_acc.append(acc_i)

            # (E) Aggregate global model
            weights_map = {cid: float(w) for cid, w in zip(client_ids, w_sel)}
            global_weights = weighted_average_weights(local_weights, client_ids, weights_map)
            global_model.load_state_dict(global_weights)

            # (F) Estimate global acc for next round
            Acc_hat_t = float(np.sum(w_sel * np.array(local_train_acc))) if len(local_train_acc) else 0.0

            # (G) Write CSV row for this epoch
            writer.writerow([
                epoch + 1,
                mean_acc,
                best10,
                worst10,
                gap_raw,
                g_norm,         # normalized unfairness signal used for alpha/r mapping
                alpha_t if bool(args.fcfl) else 0.0,
                alpha_raw if bool(args.fcfl) else 0.0,
                float(logged_r),
                var_acc,
                float(np.mean(Q)),
                float(np.std(Q)),
                float(np.min(Q)),
                float(np.max(Q)),
                int(len(idxs_users)),
                int(len(selected_by_Q)),
                int(len(selected_random)),
                Acc_hat_t
            ])

            print(f"Round {epoch+1:03d}: Mean Acc: {mean_acc*100:.2f}%, "
                  f"Best10: {best10*100:.2f}%, Worst10: {worst10*100:.2f}%, "
                  f"Gap: {gap_raw*100:.2f}%, Var: {var_acc:.6f}, "
                  f"alpha: {alpha_t:.4f}, r_used: {logged_r:.3f}, "
                  f"Q: [μ:{np.mean(Q):.4f},σ:{np.std(Q):.4f},min:{np.min(Q):.4f},max:{np.max(Q):.4f}]"
                  )


    # ---------------------------
    # Final test & run summary (minimal prints)
    # ---------------------------
    test_acc, test_loss = test_inference(args, global_model, test_dataset)

    # Final variance (recompute once at the end)
    Acc_t_final = np.zeros(N, dtype=float)
    for i in range(N):
        lu = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[i], logger=None)
        acc_i, _ = lu.inference(model=global_model)
        Acc_t_final[i] = acc_i
    final_var = float(np.var(Acc_t_final))

    # Print summary
    print("\n=== RUN SUMMARY ===")
    print(f"Log file         : {log_path}")
    print(f"Flags            : fcfl={args.fcfl}, aa={args.fcfl_adaptive_alpha}, "
          f"alpha_min={args.fcfl_alpha_min}, alpha_max={args.fcfl_alpha_max}, "
          f"r0={args.fcfl_r}, r_min={args.fcfl_r_min}, r_max={args.fcfl_r_max}, "
          f"frac={args.frac}, m={m}, users={args.num_users}, dataset={args.dataset}, model={args.model}, epochs={args.epochs}")
    print(f"Test accuracy    : {100.0*test_acc:.2f}%")
    print(f"Final variance   : {final_var:.6f}")
    print(f"Total Run Time   : {time.time() - start_time:0.4f}s")

    end_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    start_ts = datetime.fromtimestamp(start_time).strftime("%Y-%m-%d %H:%M:%S")
    duration_s = time.time() - start_time
    summary_path = os.path.join("../logs", "summary.csv")
    
    summary_row = {
        "run_log": log_path,
        "run_log_basename": os.path.basename(log_path),
        "dataset": args.dataset,
        "model": args.model,
        "num_users": args.num_users,
        "frac": args.frac,
        "m": m,
        "fcfl": int(bool(args.fcfl)),
        "fcfl_adaptive_alpha": int(bool(args.fcfl_adaptive_alpha)),
        "fcfl_r": args.fcfl_r,
        "fcfl_alpha_min": args.fcfl_alpha_min,
        "fcfl_alpha_max": args.fcfl_alpha_max,
        "epochs": args.epochs,
        "local_bs": args.local_bs,
        "local_ep": args.local_ep,
        "lr": args.lr,
        "optimizer": args.optimizer,
        "iid": args.iid,
        "test_acc": float(test_acc),
        "final_variance": float(final_var),
        "device": str(args.device),
        "start_time": start_ts,
        "end_time": end_ts,
        "duration_sec": f"{duration_s:.3f}",
    }

    os.makedirs("../logs", exist_ok=True)
    write_header = not os.path.exists(summary_path)
    with open(summary_path, "a", newline="") as fsum:
        writer = csv.DictWriter(fsum, fieldnames=list(summary_row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(summary_row)
