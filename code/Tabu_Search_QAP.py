#!/usr/bin/env python3
"""
Tabu Search for the 15-Department Quadratic Assignment Problem (QAP)
Homework 5 — Intelligent Optimization, Spring 2026

Nugent et al. Nug15 benchmark — optimal solution = 1150 (symmetric flow*distance)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import os
import json
from copy import deepcopy

# ==============================================================================
# 1. PROBLEM DATA — Nugent 15 QAP
# ==============================================================================
# Combined matrix: upper triangle = distances, lower triangle = flows, diagonal = 0
RAW_MATRIX = np.array([
    [ 0, 1, 2, 3, 4, 1, 2, 3, 4, 5, 2, 3, 4, 5, 6],
    [10, 0, 1, 2, 3, 2, 1, 2, 3, 4, 3, 2, 3, 4, 5],
    [ 0, 1, 0, 1, 2, 3, 2, 1, 2, 3, 4, 3, 2, 3, 4],
    [ 5, 3,10, 0, 1, 4, 3, 2, 1, 2, 5, 4, 3, 2, 3],
    [ 1, 2, 2, 1, 0, 5, 4, 3, 2, 1, 6, 5, 4, 3, 2],
    [ 0, 2, 0, 1, 3, 0, 1, 2, 3, 4, 1, 2, 3, 4, 5],
    [ 1, 2, 2, 5, 5, 2, 0, 1, 2, 3, 2, 1, 2, 3, 4],
    [ 2, 3, 5, 0, 5, 2, 6, 0, 1, 2, 3, 2, 1, 2, 3],
    [ 2, 2, 4, 0, 5, 1, 0, 5, 0, 1, 4, 3, 2, 1, 2],
    [ 2, 0, 5, 2, 1, 5, 1, 2, 0, 0, 5, 4, 3, 2, 1],
    [ 2, 2, 2, 1, 0, 0, 5,10,10, 0, 0, 1, 2, 3, 4],
    [ 0, 0, 2, 0, 3, 0, 5, 0, 5, 4, 5, 0, 1, 2, 3],
    [ 4,10, 5, 2, 0, 2, 5, 5,10, 0, 0, 3, 0, 1, 2],
    [ 0, 5, 5, 5, 5, 5, 1, 0, 0, 0, 5, 3,10, 0, 1],
    [ 0, 0, 5, 0, 5,10, 0, 0, 2, 5, 0, 0, 2, 4, 0],
])

N = 15  # number of departments / locations

# Extract flow matrix (lower triangle, make symmetric)
FLOW = np.zeros((N, N), dtype=int)
for i in range(N):
    for j in range(i):
        FLOW[i][j] = RAW_MATRIX[i][j]
        FLOW[j][i] = RAW_MATRIX[i][j]

# Extract distance matrix (upper triangle, make symmetric)
DIST = np.zeros((N, N), dtype=int)
for i in range(N):
    for j in range(i + 1, N):
        DIST[i][j] = RAW_MATRIX[i][j]
        DIST[j][i] = RAW_MATRIX[i][j]

OPTIMAL_COST = 1150  # known optimal for Nug15 (symmetric)


def evaluate(perm):
    """Evaluate the total flow cost for a given permutation.
    perm[i] = location assigned to department i.
    Cost = sum over all pairs (i,j) of flow[i][j] * distance[perm[i]][perm[j]]
    """
    cost = 0
    for i in range(N):
        for j in range(N):
            if i != j:
                cost += FLOW[i][j] * DIST[perm[i]][perm[j]]
    return cost


def evaluate_fast(perm):
    """Vectorized evaluation."""
    perm = np.array(perm)
    d_sub = DIST[np.ix_(perm, perm)]
    return int(np.sum(FLOW * d_sub))


# ==============================================================================
# 2. TABU SEARCH IMPLEMENTATIONS
# ==============================================================================

def tabu_search_basic(initial_perm, tabu_size=10, max_iter=500, seed=None):
    """
    Basic Tabu Search with recency-based tabu list, no aspiration.
    Move operator: pairwise swap of two departments' locations.
    Tabu entry: the pair (i, j) of swapped department indices.
    Neighborhood: complete (all C(15,2) = 105 swaps).
    """
    rng = np.random.RandomState(seed)
    n = len(initial_perm)

    current = list(initial_perm)
    current_cost = evaluate_fast(current)
    best = list(current)
    best_cost = current_cost

    tabu_list = []  # list of (i, j) pairs
    history = [best_cost]  # track best-so-far at each iteration
    current_history = [current_cost]

    for iteration in range(max_iter):
        # Generate full neighborhood: all pairwise swaps
        best_neighbor = None
        best_neighbor_cost = float('inf')
        best_swap = None

        for i in range(n - 1):
            for j in range(i + 1, n):
                # Swap departments i and j in their locations
                neighbor = list(current)
                neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
                cost = evaluate_fast(neighbor)

                swap = (i, j)
                is_tabu = swap in tabu_list

                if not is_tabu and cost < best_neighbor_cost:
                    best_neighbor = neighbor
                    best_neighbor_cost = cost
                    best_swap = swap

        # If all moves are tabu, pick the least-cost tabu move
        if best_neighbor is None:
            for i in range(n - 1):
                for j in range(i + 1, n):
                    neighbor = list(current)
                    neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
                    cost = evaluate_fast(neighbor)
                    if cost < best_neighbor_cost:
                        best_neighbor = neighbor
                        best_neighbor_cost = cost
                        best_swap = (i, j)

        # Move to best neighbor
        current = best_neighbor
        current_cost = best_neighbor_cost

        # Update tabu list
        tabu_list.append(best_swap)
        if len(tabu_list) > tabu_size:
            tabu_list.pop(0)

        # Update best
        if current_cost < best_cost:
            best = list(current)
            best_cost = current_cost

        history.append(best_cost)
        current_history.append(current_cost)

    return best, best_cost, history, current_history


def tabu_search_dynamic(initial_perm, tabu_range=(5, 15), change_every=20,
                        max_iter=500, seed=None):
    """
    Tabu Search with dynamic tabu list size (random uniform integer in tabu_range,
    changed every `change_every` iterations).
    """
    rng = np.random.RandomState(seed)
    n = len(initial_perm)

    current = list(initial_perm)
    current_cost = evaluate_fast(current)
    best = list(current)
    best_cost = current_cost

    tabu_list = []
    tabu_size = rng.randint(tabu_range[0], tabu_range[1] + 1)
    history = [best_cost]
    current_history = [current_cost]

    for iteration in range(max_iter):
        # Dynamically adjust tabu size
        if iteration > 0 and iteration % change_every == 0:
            tabu_size = rng.randint(tabu_range[0], tabu_range[1] + 1)

        best_neighbor = None
        best_neighbor_cost = float('inf')
        best_swap = None

        for i in range(n - 1):
            for j in range(i + 1, n):
                neighbor = list(current)
                neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
                cost = evaluate_fast(neighbor)
                swap = (i, j)
                is_tabu = swap in tabu_list

                if not is_tabu and cost < best_neighbor_cost:
                    best_neighbor = neighbor
                    best_neighbor_cost = cost
                    best_swap = swap

        if best_neighbor is None:
            for i in range(n - 1):
                for j in range(i + 1, n):
                    neighbor = list(current)
                    neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
                    cost = evaluate_fast(neighbor)
                    if cost < best_neighbor_cost:
                        best_neighbor = neighbor
                        best_neighbor_cost = cost
                        best_swap = (i, j)

        current = best_neighbor
        current_cost = best_neighbor_cost

        tabu_list.append(best_swap)
        if len(tabu_list) > tabu_size:
            tabu_list.pop(0)

        if current_cost < best_cost:
            best = list(current)
            best_cost = current_cost

        history.append(best_cost)
        current_history.append(current_cost)

    return best, best_cost, history, current_history


def tabu_search_aspiration(initial_perm, tabu_size=10, max_iter=500, seed=None):
    """
    Tabu Search with aspiration criterion: a tabu move is allowed if it
    produces a solution better than the best-so-far.
    """
    rng = np.random.RandomState(seed)
    n = len(initial_perm)

    current = list(initial_perm)
    current_cost = evaluate_fast(current)
    best = list(current)
    best_cost = current_cost

    tabu_list = []
    history = [best_cost]
    current_history = [current_cost]

    for iteration in range(max_iter):
        best_neighbor = None
        best_neighbor_cost = float('inf')
        best_swap = None

        for i in range(n - 1):
            for j in range(i + 1, n):
                neighbor = list(current)
                neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
                cost = evaluate_fast(neighbor)
                swap = (i, j)
                is_tabu = swap in tabu_list

                # Accept if not tabu, OR if aspiration criterion met (better than best)
                if (not is_tabu or cost < best_cost) and cost < best_neighbor_cost:
                    best_neighbor = neighbor
                    best_neighbor_cost = cost
                    best_swap = swap

        if best_neighbor is None:
            # Fallback: pick least-cost move regardless
            for i in range(n - 1):
                for j in range(i + 1, n):
                    neighbor = list(current)
                    neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
                    cost = evaluate_fast(neighbor)
                    if cost < best_neighbor_cost:
                        best_neighbor = neighbor
                        best_neighbor_cost = cost
                        best_swap = (i, j)

        current = best_neighbor
        current_cost = best_neighbor_cost

        tabu_list.append(best_swap)
        if len(tabu_list) > tabu_size:
            tabu_list.pop(0)

        if current_cost < best_cost:
            best = list(current)
            best_cost = current_cost

        history.append(best_cost)
        current_history.append(current_cost)

    return best, best_cost, history, current_history


def tabu_search_partial(initial_perm, tabu_size=10, max_iter=500, seed=None,
                        sample_frac=0.5):
    """
    Tabu Search with aspiration + partial neighborhood (randomly sample 50%
    of the C(15,2) possible swaps each iteration).
    """
    rng = np.random.RandomState(seed)
    n = len(initial_perm)

    current = list(initial_perm)
    current_cost = evaluate_fast(current)
    best = list(current)
    best_cost = current_cost

    tabu_list = []
    history = [best_cost]
    current_history = [current_cost]

    # Pre-generate all swap pairs
    all_swaps = [(i, j) for i in range(n - 1) for j in range(i + 1, n)]
    num_sample = max(1, int(len(all_swaps) * sample_frac))

    for iteration in range(max_iter):
        # Randomly sample a subset of the neighborhood
        sampled = rng.choice(len(all_swaps), size=num_sample, replace=False)

        best_neighbor = None
        best_neighbor_cost = float('inf')
        best_swap = None

        for idx in sampled:
            i, j = all_swaps[idx]
            neighbor = list(current)
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
            cost = evaluate_fast(neighbor)
            swap = (i, j)
            is_tabu = swap in tabu_list

            if (not is_tabu or cost < best_cost) and cost < best_neighbor_cost:
                best_neighbor = neighbor
                best_neighbor_cost = cost
                best_swap = swap

        if best_neighbor is None:
            # If all sampled moves are tabu, pick least-cost from sample
            for idx in sampled:
                i, j = all_swaps[idx]
                neighbor = list(current)
                neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
                cost = evaluate_fast(neighbor)
                if cost < best_neighbor_cost:
                    best_neighbor = neighbor
                    best_neighbor_cost = cost
                    best_swap = (i, j)

        current = best_neighbor
        current_cost = best_neighbor_cost

        tabu_list.append(best_swap)
        if len(tabu_list) > tabu_size:
            tabu_list.pop(0)

        if current_cost < best_cost:
            best = list(current)
            best_cost = current_cost

        history.append(best_cost)
        current_history.append(current_cost)

    return best, best_cost, history, current_history


def tabu_search_diversification(initial_perm, tabu_size=10, max_iter=500,
                                seed=None, stagnation_limit=30,
                                strategy="restart"):
    """
    Tabu Search with aspiration + diversification mechanism.
    strategy="restart": random restart after stagnation_limit iterations without improvement.
    strategy="frequency": frequency-based penalty added to cost.
    """
    rng = np.random.RandomState(seed)
    n = len(initial_perm)

    current = list(initial_perm)
    current_cost = evaluate_fast(current)
    best = list(current)
    best_cost = current_cost

    tabu_list = []
    history = [best_cost]
    current_history = [current_cost]
    stagnation_count = 0

    # Frequency matrix: how often each department is placed at each location
    freq_matrix = np.zeros((n, n), dtype=int)
    for dept in range(n):
        freq_matrix[dept][current[dept]] += 1

    for iteration in range(max_iter):
        # Check for diversification trigger
        if stagnation_count >= stagnation_limit:
            if strategy == "restart":
                # Random restart: generate a completely new random permutation
                current = list(rng.permutation(n))
                current_cost = evaluate_fast(current)
                tabu_list = []  # clear tabu list on restart
                stagnation_count = 0
            elif strategy == "frequency":
                # Frequency penalty: heavily penalize frequently visited positions
                # Find the most frequently visited dept-location pair and force a swap away
                max_freq_dept = np.argmax(np.max(freq_matrix, axis=1))
                max_freq_loc = np.argmax(freq_matrix[max_freq_dept])
                # Swap this department with a random other
                other_dept = rng.randint(0, n)
                while other_dept == max_freq_dept:
                    other_dept = rng.randint(0, n)
                current[max_freq_dept], current[other_dept] = current[other_dept], current[max_freq_dept]
                current_cost = evaluate_fast(current)
                stagnation_count = 0

        best_neighbor = None
        best_neighbor_cost = float('inf')
        best_swap = None

        for i in range(n - 1):
            for j in range(i + 1, n):
                neighbor = list(current)
                neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
                cost = evaluate_fast(neighbor)

                # Add frequency penalty for diversification
                if strategy == "frequency":
                    penalty = 0
                    lam = 0.5  # penalty weight
                    for dept in range(n):
                        penalty += freq_matrix[dept][neighbor[dept]]
                    cost_adj = cost + lam * penalty
                else:
                    cost_adj = cost

                swap = (i, j)
                is_tabu = swap in tabu_list

                if (not is_tabu or cost < best_cost) and cost_adj < best_neighbor_cost:
                    best_neighbor = neighbor
                    best_neighbor_cost = cost_adj
                    best_swap = swap

        if best_neighbor is None:
            for i in range(n - 1):
                for j in range(i + 1, n):
                    neighbor = list(current)
                    neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
                    cost = evaluate_fast(neighbor)
                    if cost < best_neighbor_cost:
                        best_neighbor = neighbor
                        best_neighbor_cost = cost
                        best_swap = (i, j)

        actual_cost = evaluate_fast(best_neighbor)
        current = best_neighbor
        current_cost = actual_cost

        # Update frequency matrix
        for dept in range(n):
            freq_matrix[dept][current[dept]] += 1

        tabu_list.append(best_swap)
        if len(tabu_list) > tabu_size:
            tabu_list.pop(0)

        if current_cost < best_cost:
            best = list(current)
            best_cost = current_cost
            stagnation_count = 0
        else:
            stagnation_count += 1

        history.append(best_cost)
        current_history.append(current_cost)

    return best, best_cost, history, current_history


# ==============================================================================
# 3. RUN ALL 35 EXPERIMENTS
# ==============================================================================

def generate_random_perm(seed):
    """Generate a random permutation of 0..14 using a given seed."""
    rng = np.random.RandomState(seed)
    return list(rng.permutation(N))


# Fixed 5 seeds for all experiments
SEEDS = [42, 123, 256, 789, 1024]
MAX_ITER = 500
BASE_TABU_SIZE = 10  # original tabu list size

results_all = []
run_id = 0
all_histories = {}

print("=" * 80)
print("TABU SEARCH FOR 15-DEPARTMENT QAP (Nugent Nug15)")
print(f"Known Optimal = {OPTIMAL_COST}")
print("=" * 80)

# ---------- Experiment 1: Basic TS (tabu_size=10, no aspiration) — 5 runs ----------
print("\n" + "=" * 70)
print("EXPERIMENT 1: Basic TS — tabu size=10, full neighborhood, no aspiration")
print("=" * 70)
for seed in SEEDS:
    run_id += 1
    init_perm = generate_random_perm(seed)
    init_cost = evaluate_fast(init_perm)
    t0 = time.perf_counter()
    best_perm, best_cost, hist, curr_hist = tabu_search_basic(
        init_perm, tabu_size=BASE_TABU_SIZE, max_iter=MAX_ITER, seed=seed
    )
    elapsed = time.perf_counter() - t0
    results_all.append({
        'run_id': run_id, 'experiment': 'E1_basic', 'seed': seed,
        'tabu_size': BASE_TABU_SIZE, 'neighborhood': 'complete',
        'aspiration': 'none', 'diversification': 'none',
        'init_cost': init_cost, 'best_cost': best_cost,
        'best_perm': best_perm, 'runtime_s': elapsed
    })
    all_histories[run_id] = hist
    print(f"  Seed {seed:>5}: init={init_cost}, best={best_cost}, "
          f"gap={best_cost - OPTIMAL_COST}, time={elapsed:.2f}s")

# ---------- Experiment 2: Smaller tabu size=5 — 5 runs ----------
print("\n" + "=" * 70)
print("EXPERIMENT 2: Smaller tabu size=5")
print("=" * 70)
SMALL_TABU = 5
for seed in SEEDS:
    run_id += 1
    init_perm = generate_random_perm(seed)
    init_cost = evaluate_fast(init_perm)
    t0 = time.perf_counter()
    best_perm, best_cost, hist, curr_hist = tabu_search_basic(
        init_perm, tabu_size=SMALL_TABU, max_iter=MAX_ITER, seed=seed
    )
    elapsed = time.perf_counter() - t0
    results_all.append({
        'run_id': run_id, 'experiment': 'E2_small_tabu', 'seed': seed,
        'tabu_size': SMALL_TABU, 'neighborhood': 'complete',
        'aspiration': 'none', 'diversification': 'none',
        'init_cost': init_cost, 'best_cost': best_cost,
        'best_perm': best_perm, 'runtime_s': elapsed
    })
    all_histories[run_id] = hist
    print(f"  Seed {seed:>5}: init={init_cost}, best={best_cost}, "
          f"gap={best_cost - OPTIMAL_COST}, time={elapsed:.2f}s")

# ---------- Experiment 3: Larger tabu size=15 — 5 runs ----------
print("\n" + "=" * 70)
print("EXPERIMENT 3: Larger tabu size=15")
print("=" * 70)
LARGE_TABU = 15
for seed in SEEDS:
    run_id += 1
    init_perm = generate_random_perm(seed)
    init_cost = evaluate_fast(init_perm)
    t0 = time.perf_counter()
    best_perm, best_cost, hist, curr_hist = tabu_search_basic(
        init_perm, tabu_size=LARGE_TABU, max_iter=MAX_ITER, seed=seed
    )
    elapsed = time.perf_counter() - t0
    results_all.append({
        'run_id': run_id, 'experiment': 'E3_large_tabu', 'seed': seed,
        'tabu_size': LARGE_TABU, 'neighborhood': 'complete',
        'aspiration': 'none', 'diversification': 'none',
        'init_cost': init_cost, 'best_cost': best_cost,
        'best_perm': best_perm, 'runtime_s': elapsed
    })
    all_histories[run_id] = hist
    print(f"  Seed {seed:>5}: init={init_cost}, best={best_cost}, "
          f"gap={best_cost - OPTIMAL_COST}, time={elapsed:.2f}s")

# ---------- Experiment 4: Dynamic tabu size — 5 runs ----------
print("\n" + "=" * 70)
print("EXPERIMENT 4: Dynamic tabu size, Uniform(5, 15), change every 20 iter")
print("=" * 70)
for seed in SEEDS:
    run_id += 1
    init_perm = generate_random_perm(seed)
    init_cost = evaluate_fast(init_perm)
    t0 = time.perf_counter()
    best_perm, best_cost, hist, curr_hist = tabu_search_dynamic(
        init_perm, tabu_range=(5, 15), change_every=20,
        max_iter=MAX_ITER, seed=seed
    )
    elapsed = time.perf_counter() - t0
    results_all.append({
        'run_id': run_id, 'experiment': 'E4_dynamic_tabu', 'seed': seed,
        'tabu_size': 'dynamic(5-15)', 'neighborhood': 'complete',
        'aspiration': 'none', 'diversification': 'none',
        'init_cost': init_cost, 'best_cost': best_cost,
        'best_perm': best_perm, 'runtime_s': elapsed
    })
    all_histories[run_id] = hist
    print(f"  Seed {seed:>5}: init={init_cost}, best={best_cost}, "
          f"gap={best_cost - OPTIMAL_COST}, time={elapsed:.2f}s")

# ---------- Experiment 5: Aspiration (best-so-far) + tabu_size=10 — 5 runs ----------
print("\n" + "=" * 70)
print("EXPERIMENT 5: Aspiration criterion (best-so-far), tabu size=10")
print("=" * 70)
ASPIRATION_TABU = 10
for seed in SEEDS:
    run_id += 1
    init_perm = generate_random_perm(seed)
    init_cost = evaluate_fast(init_perm)
    t0 = time.perf_counter()
    best_perm, best_cost, hist, curr_hist = tabu_search_aspiration(
        init_perm, tabu_size=ASPIRATION_TABU, max_iter=MAX_ITER, seed=seed
    )
    elapsed = time.perf_counter() - t0
    results_all.append({
        'run_id': run_id, 'experiment': 'E5_aspiration', 'seed': seed,
        'tabu_size': ASPIRATION_TABU, 'neighborhood': 'complete',
        'aspiration': 'best_so_far', 'diversification': 'none',
        'init_cost': init_cost, 'best_cost': best_cost,
        'best_perm': best_perm, 'runtime_s': elapsed
    })
    all_histories[run_id] = hist
    print(f"  Seed {seed:>5}: init={init_cost}, best={best_cost}, "
          f"gap={best_cost - OPTIMAL_COST}, time={elapsed:.2f}s")

# ---------- Experiment 6: Partial neighborhood (50%) + aspiration — 5 runs ----------
print("\n" + "=" * 70)
print("EXPERIMENT 6: Partial neighborhood (50%) + aspiration, tabu size=10")
print("=" * 70)
for seed in SEEDS:
    run_id += 1
    init_perm = generate_random_perm(seed)
    init_cost = evaluate_fast(init_perm)
    t0 = time.perf_counter()
    best_perm, best_cost, hist, curr_hist = tabu_search_partial(
        init_perm, tabu_size=ASPIRATION_TABU, max_iter=MAX_ITER, seed=seed,
        sample_frac=0.5
    )
    elapsed = time.perf_counter() - t0
    results_all.append({
        'run_id': run_id, 'experiment': 'E6_partial', 'seed': seed,
        'tabu_size': ASPIRATION_TABU, 'neighborhood': 'partial_50pct',
        'aspiration': 'best_so_far', 'diversification': 'none',
        'init_cost': init_cost, 'best_cost': best_cost,
        'best_perm': best_perm, 'runtime_s': elapsed
    })
    all_histories[run_id] = hist
    print(f"  Seed {seed:>5}: init={init_cost}, best={best_cost}, "
          f"gap={best_cost - OPTIMAL_COST}, time={elapsed:.2f}s")

# ---------- Experiment 7: Diversification (random restart) + aspiration — 5 runs ----------
print("\n" + "=" * 70)
print("EXPERIMENT 7: Random restart diversification + aspiration, tabu size=10")
print("=" * 70)
for seed in SEEDS:
    run_id += 1
    init_perm = generate_random_perm(seed)
    init_cost = evaluate_fast(init_perm)
    t0 = time.perf_counter()
    best_perm, best_cost, hist, curr_hist = tabu_search_diversification(
        init_perm, tabu_size=ASPIRATION_TABU, max_iter=MAX_ITER, seed=seed,
        stagnation_limit=30, strategy="restart"
    )
    elapsed = time.perf_counter() - t0
    results_all.append({
        'run_id': run_id, 'experiment': 'E7_diversification', 'seed': seed,
        'tabu_size': ASPIRATION_TABU, 'neighborhood': 'complete',
        'aspiration': 'best_so_far', 'diversification': 'random_restart',
        'init_cost': init_cost, 'best_cost': best_cost,
        'best_perm': best_perm, 'runtime_s': elapsed
    })
    all_histories[run_id] = hist
    print(f"  Seed {seed:>5}: init={init_cost}, best={best_cost}, "
          f"gap={best_cost - OPTIMAL_COST}, time={elapsed:.2f}s")


# ==============================================================================
# 4. SAVE RESULTS
# ==============================================================================

results_df = pd.DataFrame(results_all)
os.makedirs('TS_Results', exist_ok=True)

# Save CSV (without perm lists for clean CSV)
save_cols = [c for c in results_df.columns if c != 'best_perm']
results_df[save_cols].to_csv('TS_Results/all_35_runs.csv', index=False)

# Save histories as JSON
with open('TS_Results/histories.json', 'w') as f:
    json.dump({str(k): v for k, v in all_histories.items()}, f)

# Save full results with perms
results_df['best_perm_str'] = results_df['best_perm'].apply(str)
results_df.drop(columns=['best_perm']).to_csv('TS_Results/all_35_runs_full.csv', index=False)

print("\n" + "=" * 80)
print("ALL 35 RUNS COMPLETE")
print("=" * 80)
print(f"\nOverall Best Cost: {results_df['best_cost'].min()}")
print(f"Optimal: {OPTIMAL_COST}")
print(f"Mean Cost: {results_df['best_cost'].mean():.1f}")


# ==============================================================================
# 5. ANALYSIS & PLOTS
# ==============================================================================

# Color palette
COLORS = {
    'E1_basic': '#2E86AB',
    'E2_small_tabu': '#A23B72',
    'E3_large_tabu': '#F18F01',
    'E4_dynamic_tabu': '#C73E1D',
    'E5_aspiration': '#3B1F2B',
    'E6_partial': '#44BBA4',
    'E7_diversification': '#E94F37',
}

EXP_LABELS = {
    'E1_basic': 'E1: Basic TS\n(t=10)',
    'E2_small_tabu': 'E2: Small Tabu\n(t=5)',
    'E3_large_tabu': 'E3: Large Tabu\n(t=15)',
    'E4_dynamic_tabu': 'E4: Dynamic Tabu\n(t=5-15)',
    'E5_aspiration': 'E5: Aspiration\n(best-so-far)',
    'E6_partial': 'E6: Partial\nNeighborhood',
    'E7_diversification': 'E7: Diversification\n(restart)',
}

EXP_LABELS_SHORT = {
    'E1_basic': 'E1: Basic',
    'E2_small_tabu': 'E2: Small (t=5)',
    'E3_large_tabu': 'E3: Large (t=15)',
    'E4_dynamic_tabu': 'E4: Dynamic',
    'E5_aspiration': 'E5: Aspiration',
    'E6_partial': 'E6: Partial NH',
    'E7_diversification': 'E7: Restart',
}

experiments = ['E1_basic', 'E2_small_tabu', 'E3_large_tabu', 'E4_dynamic_tabu',
               'E5_aspiration', 'E6_partial', 'E7_diversification']

plt.rcParams.update({
    'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 12,
    'figure.dpi': 150, 'savefig.dpi': 150, 'savefig.bbox': 'tight'
})

# --- Plot 1: Boxplot of best costs by experiment ---
fig, ax = plt.subplots(figsize=(12, 6))
data_box = [results_df[results_df['experiment'] == e]['best_cost'].values for e in experiments]
bp = ax.boxplot(data_box, labels=[EXP_LABELS[e] for e in experiments],
                patch_artist=True, widths=0.6)
for patch, exp in zip(bp['boxes'], experiments):
    patch.set_facecolor(COLORS[exp])
    patch.set_alpha(0.7)
ax.axhline(OPTIMAL_COST, color='red', linestyle='--', linewidth=1.5, label=f'Optimal ({OPTIMAL_COST})')
ax.set_ylabel('Best Cost Found')
#ax.set_title('Best Cost by TS Variant (All 35 Runs)')  # caption in report
ax.legend(loc='upper right')
plt.tight_layout()
plt.savefig('TS_Results/fig1_boxplot_all_experiments.png')
plt.close()
print("Saved: fig1_boxplot_all_experiments.png")

# --- Plot 2: Convergence curves for preliminary runs (E1) ---
fig, ax = plt.subplots(figsize=(10, 6))
e1_runs = results_df[results_df['experiment'] == 'E1_basic']
for _, row in e1_runs.iterrows():
    hist = all_histories[row['run_id']]
    ax.plot(hist, linewidth=1.5, label=f"Seed {row['seed']}")
ax.axhline(OPTIMAL_COST, color='red', linestyle='--', linewidth=1.2, label=f'Optimal ({OPTIMAL_COST})')
ax.set_xlabel('Iteration')
ax.set_ylabel('Best Cost So Far')
#ax.set_title('Experiment 1: Convergence of Basic TS (5 Seeds)')  # caption in report
ax.legend()
plt.tight_layout()
plt.savefig('TS_Results/fig2_convergence_basic.png')
plt.close()
print("Saved: fig2_convergence_basic.png")

# --- Plot 3: Bar chart — near-optimums by seed for E1 ---
fig, ax = plt.subplots(figsize=(8, 5))
e1_seeds = e1_runs['seed'].values
e1_costs = e1_runs['best_cost'].values
bars = ax.bar(range(len(e1_seeds)), e1_costs, color=COLORS['E1_basic'], alpha=0.8, edgecolor='black')
ax.axhline(OPTIMAL_COST, color='red', linestyle='--', linewidth=1.5, label=f'Optimal ({OPTIMAL_COST})')
ax.axhline(np.mean(e1_costs), color='orange', linestyle='-', linewidth=1.5, label=f'Mean ({np.mean(e1_costs):.0f})')
ax.set_xticks(range(len(e1_seeds)))
ax.set_xticklabels([f'Seed {s}' for s in e1_seeds])
ax.set_ylabel('Best Cost')
#ax.set_title('Experiment 1: Near-Optimums by Seed')  # caption in report
ax.legend()
plt.tight_layout()
plt.savefig('TS_Results/fig3_bar_basic_seeds.png')
plt.close()
print("Saved: fig3_bar_basic_seeds.png")

# --- Plot 4: Boxplot comparing tabu sizes (E1, E2, E3, E4) ---
fig, ax = plt.subplots(figsize=(9, 6))
tabu_exps = ['E2_small_tabu', 'E1_basic', 'E3_large_tabu', 'E4_dynamic_tabu']
tabu_labels = ['t=5', 't=10', 't=15', 'Dynamic\n(5-15)']
data_tabu = [results_df[results_df['experiment'] == e]['best_cost'].values for e in tabu_exps]
bp2 = ax.boxplot(data_tabu, labels=tabu_labels, patch_artist=True, widths=0.5)
for patch, exp in zip(bp2['boxes'], tabu_exps):
    patch.set_facecolor(COLORS[exp])
    patch.set_alpha(0.7)
ax.axhline(OPTIMAL_COST, color='red', linestyle='--', linewidth=1.5, label=f'Optimal ({OPTIMAL_COST})')
ax.set_ylabel('Best Cost Found')
#ax.set_title('Effect of Tabu List Size on Performance')  # caption in report
ax.legend()
plt.tight_layout()
plt.savefig('TS_Results/fig4_boxplot_tabu_sizes.png')
plt.close()
print("Saved: fig4_boxplot_tabu_sizes.png")

# --- Plot 5: Convergence curves for tabu size comparison (one seed) ---
fig, ax = plt.subplots(figsize=(10, 6))
for exp_name in tabu_exps:
    row = results_df[(results_df['experiment'] == exp_name) & (results_df['seed'] == SEEDS[0])].iloc[0]
    hist = all_histories[row['run_id']]
    ax.plot(hist[:200], linewidth=1.5, label=EXP_LABELS_SHORT[exp_name], color=COLORS[exp_name])
ax.axhline(OPTIMAL_COST, color='red', linestyle='--', linewidth=1.2, label=f'Optimal ({OPTIMAL_COST})')
ax.set_xlabel('Iteration')
ax.set_ylabel('Best Cost So Far')
#ax.set_title(f'Tabu Size Comparison — Convergence (Seed={SEEDS[0]})')  # caption in report
ax.legend()
plt.tight_layout()
plt.savefig('TS_Results/fig5_convergence_tabu_sizes.png')
plt.close()
print("Saved: fig5_convergence_tabu_sizes.png")

# --- Plot 6: Boxplot — Aspiration vs No Aspiration ---
fig, ax = plt.subplots(figsize=(7, 5))
asp_exps = ['E1_basic', 'E5_aspiration']
asp_labels = ['No Aspiration\n(E1: t=10)', 'Aspiration\n(E5: best-so-far)']
data_asp = [results_df[results_df['experiment'] == e]['best_cost'].values for e in asp_exps]
bp3 = ax.boxplot(data_asp, labels=asp_labels, patch_artist=True, widths=0.4)
for patch, exp in zip(bp3['boxes'], asp_exps):
    patch.set_facecolor(COLORS[exp])
    patch.set_alpha(0.7)
ax.axhline(OPTIMAL_COST, color='red', linestyle='--', linewidth=1.5, label=f'Optimal ({OPTIMAL_COST})')
ax.set_ylabel('Best Cost Found')
#ax.set_title('Effect of Aspiration Criterion')  # caption in report
ax.legend()
plt.tight_layout()
plt.savefig('TS_Results/fig6_boxplot_aspiration.png')
plt.close()
print("Saved: fig6_boxplot_aspiration.png")

# --- Plot 7: Boxplot — Partial Neighborhood vs Full ---
fig, ax = plt.subplots(figsize=(7, 5))
partial_exps = ['E5_aspiration', 'E6_partial']
partial_labels = ['Full Neighborhood\n(E5)', 'Partial 50%\n(E6)']
data_part = [results_df[results_df['experiment'] == e]['best_cost'].values for e in partial_exps]
bp4 = ax.boxplot(data_part, labels=partial_labels, patch_artist=True, widths=0.4)
for patch, exp in zip(bp4['boxes'], partial_exps):
    patch.set_facecolor(COLORS[exp])
    patch.set_alpha(0.7)
ax.axhline(OPTIMAL_COST, color='red', linestyle='--', linewidth=1.5, label=f'Optimal ({OPTIMAL_COST})')
ax.set_ylabel('Best Cost Found')
#ax.set_title('Effect of Partial Neighborhood Exploration')  # caption in report
ax.legend()
plt.tight_layout()
plt.savefig('TS_Results/fig7_boxplot_partial.png')
plt.close()
print("Saved: fig7_boxplot_partial.png")

# --- Plot 8: Boxplot — Diversification comparison ---
fig, ax = plt.subplots(figsize=(8, 5))
div_exps = ['E5_aspiration', 'E7_diversification']
div_labels = ['No Diversification\n(E5: Aspiration only)', 'Random Restart\n(E7: + Diversification)']
data_div = [results_df[results_df['experiment'] == e]['best_cost'].values for e in div_exps]
bp5 = ax.boxplot(data_div, labels=div_labels, patch_artist=True, widths=0.4)
for patch, exp in zip(bp5['boxes'], div_exps):
    patch.set_facecolor(COLORS[exp])
    patch.set_alpha(0.7)
ax.axhline(OPTIMAL_COST, color='red', linestyle='--', linewidth=1.5, label=f'Optimal ({OPTIMAL_COST})')
ax.set_ylabel('Best Cost Found')
#ax.set_title('Effect of Diversification (Random Restart)')  # caption in report
ax.legend()
plt.tight_layout()
plt.savefig('TS_Results/fig8_boxplot_diversification.png')
plt.close()
print("Saved: fig8_boxplot_diversification.png")

# --- Plot 9: Combined convergence (all experiments, one seed) ---
fig, ax = plt.subplots(figsize=(12, 6))
for exp_name in experiments:
    row = results_df[(results_df['experiment'] == exp_name) & (results_df['seed'] == SEEDS[0])].iloc[0]
    hist = all_histories[row['run_id']]
    ax.plot(hist[:200], linewidth=1.5, label=EXP_LABELS_SHORT[exp_name], color=COLORS[exp_name])
ax.axhline(OPTIMAL_COST, color='red', linestyle='--', linewidth=1.2, label=f'Optimal ({OPTIMAL_COST})')
ax.set_xlabel('Iteration')
ax.set_ylabel('Best Cost So Far')
#ax.set_title(f'Convergence Comparison Across All Variants (Seed={SEEDS[0]})')  # caption in report
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig('TS_Results/fig9_convergence_all.png')
plt.close()
print("Saved: fig9_convergence_all.png")

# --- Plot 10: Summary statistics table as image ---
summary_data = []
for exp in experiments:
    sub = results_df[results_df['experiment'] == exp]
    summary_data.append({
        'Experiment': EXP_LABELS_SHORT[exp],
        'Runs': len(sub),
        'Mean': f"{sub['best_cost'].mean():.1f}",
        'Std': f"{sub['best_cost'].std():.1f}",
        'Min': f"{sub['best_cost'].min()}",
        'Max': f"{sub['best_cost'].max()}",
        'Avg Runtime': f"{sub['runtime_s'].mean():.2f}s"
    })
summary_df = pd.DataFrame(summary_data)
print("\n" + "=" * 80)
print("SUMMARY TABLE")
print("=" * 80)
print(summary_df.to_string(index=False))

# Save summary CSV
summary_df.to_csv('TS_Results/summary_table.csv', index=False)

# --- Print best overall solution ---
best_row = results_df.loc[results_df['best_cost'].idxmin()]
print(f"\n{'='*80}")
print("BEST SOLUTION FOUND ACROSS ALL 35 RUNS")
print(f"{'='*80}")
print(f"  Experiment:   {best_row['experiment']}")
print(f"  Seed:         {best_row['seed']}")
print(f"  Best Cost:    {best_row['best_cost']}")
print(f"  Optimal:      {OPTIMAL_COST}")
print(f"  Gap:          {best_row['best_cost'] - OPTIMAL_COST}")
print(f"  Permutation:  {best_row['best_perm']}")
print(f"  Runtime:      {best_row['runtime_s']:.2f}s")

print("\nAll plots and data saved to TS_Results/")

# ==============================================================================
# 6. ENHANCED ANALYSIS — Log-scale convergence, heatmap, stats
# ==============================================================================
from scipy.stats import f_oneway, ttest_ind
import re

# --- Log-scale convergence: E1 (gap from optimal) ---
fig, ax = plt.subplots(figsize=(10, 6))
e1_runs = results_df[results_df['experiment'] == 'E1_basic']
for _, row in e1_runs.iterrows():
    hist = all_histories[row['run_id']]
    gap = [max(h - OPTIMAL_COST, 0.5) for h in hist]
    ax.plot(gap, linewidth=1.5, label=f"Seed {int(row['seed'])}")
ax.set_yscale('log')
ax.set_xlabel('Iteration')
ax.set_ylabel('Gap from Optimal (log scale)')
#ax.set_title('Experiment 1: Convergence Gap from Optimal (5 Seeds)')  # caption in report
ax.axhline(1, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Gap = 0')
ax.legend()
plt.tight_layout()
plt.savefig('TS_Results/fig2_convergence_basic.png')
plt.close()

# --- Log-scale convergence: tabu size comparison ---
fig, ax = plt.subplots(figsize=(10, 6))
tabu_exps = ['E2_small_tabu', 'E1_basic', 'E3_large_tabu', 'E4_dynamic_tabu']
for exp_name in tabu_exps:
    row = results_df[(results_df['experiment'] == exp_name) & (results_df['seed'] == SEEDS[0])].iloc[0]
    hist = all_histories[row['run_id']]
    gap = [max(h - OPTIMAL_COST, 0.5) for h in hist]
    ax.plot(gap, linewidth=1.5, label=EXP_LABELS_SHORT[exp_name], color=COLORS[exp_name])
ax.set_yscale('log')
ax.set_xlabel('Iteration')
ax.set_ylabel('Gap from Optimal (log scale)')
#ax.set_title(f'Tabu Size Comparison — Convergence Gap (Seed={SEEDS[0]})')  # caption in report
ax.legend()
plt.tight_layout()
plt.savefig('TS_Results/fig5_convergence_tabu_sizes.png')
plt.close()

# --- Log-scale convergence: all variants ---
fig, ax = plt.subplots(figsize=(12, 6))
for exp_name in experiments:
    row = results_df[(results_df['experiment'] == exp_name) & (results_df['seed'] == SEEDS[0])].iloc[0]
    hist = all_histories[row['run_id']]
    gap = [max(h - OPTIMAL_COST, 0.5) for h in hist]
    ax.plot(gap, linewidth=1.5, label=EXP_LABELS_SHORT[exp_name], color=COLORS[exp_name])
ax.set_yscale('log')
ax.set_xlabel('Iteration')
ax.set_ylabel('Gap from Optimal (log scale)')
#ax.set_title(f'Convergence Comparison — All TS Variants (Seed={SEEDS[0]}, Log Scale)')  # caption in report
ax.legend(fontsize=9)
ax.set_xlim(0, 300)
plt.tight_layout()
plt.savefig('TS_Results/fig9_convergence_all.png')
plt.close()

# --- Convergence subplots per experiment (all seeds, log gap) ---
fig, axes = plt.subplots(2, 4, figsize=(18, 9))
axes = axes.flatten()
for idx, exp in enumerate(experiments):
    ax = axes[idx]
    subset = results_df[results_df['experiment'] == exp]
    for _, row in subset.iterrows():
        hist = all_histories[row['run_id']]
        gap = [max(h - OPTIMAL_COST, 0.5) for h in hist]
        ax.plot(gap, alpha=0.7, linewidth=1.2, label=f"s={int(row['seed'])}")
    ax.set_yscale('log')
    #ax.set_title(EXP_LABELS_SHORT[exp], fontsize=11)  # caption in report
    ax.set_xlabel('Iteration', fontsize=9)
    ax.set_ylabel('Gap', fontsize=9)
    ax.legend(fontsize=7, loc='upper right')
    ax.set_xlim(0, 300)
axes[7].set_visible(False)
#fig.suptitle('Convergence Curves by Experiment (All Seeds, Log Scale)', fontsize=14, y=1.01)  # caption in report
plt.tight_layout()
plt.savefig('TS_Results/fig10_convergence_subplots.png')
plt.close()

# --- Heatmap: best cost by experiment × seed ---
fig, ax = plt.subplots(figsize=(10, 5))
pivot = results_df.pivot_table(index='experiment', columns='seed', values='best_cost')
pivot = pivot.reindex(experiments)
pivot.index = [EXP_LABELS_SHORT[e] for e in experiments]
im = ax.imshow(pivot.values, cmap='RdYlGn_r', aspect='auto', vmin=1150, vmax=1190)
ax.set_xticks(range(len(SEEDS)))
ax.set_xticklabels([f'Seed {s}' for s in SEEDS])
ax.set_yticks(range(len(experiments)))
ax.set_yticklabels(pivot.index)
for i in range(pivot.shape[0]):
    for j in range(pivot.shape[1]):
        val = int(pivot.values[i, j])
        color = 'white' if val > 1170 else 'black'
        fontweight = 'bold' if val == 1150 else 'normal'
        ax.text(j, i, str(val), ha='center', va='center', fontsize=11,
                color=color, fontweight=fontweight)
cbar = plt.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label('Best Cost')
#ax.set_title('Best Cost Heatmap: Experiment × Seed')  # caption in report
plt.tight_layout()
plt.savefig('TS_Results/fig11_heatmap_cost.png')
plt.close()

# --- Success rate bar chart ---
fig, ax = plt.subplots(figsize=(10, 5))
opt_pcts = []
for exp in experiments:
    costs = results_df[results_df['experiment'] == exp]['best_cost'].values
    pct = 100 * np.sum(costs == OPTIMAL_COST) / len(costs)
    opt_pcts.append(pct)
bars = ax.bar(range(len(experiments)), opt_pcts,
              color=[COLORS[e] for e in experiments], edgecolor='black', alpha=0.85)
ax.set_xticks(range(len(experiments)))
ax.set_xticklabels([EXP_LABELS_SHORT[e] for e in experiments], rotation=20, ha='right')
ax.set_ylabel('% Runs Reaching Optimal (1150)')
#ax.set_title('Success Rate: Percentage of Runs Reaching Global Optimum')  # caption in report
ax.set_ylim(0, 105)
for bar, pct in zip(bars, opt_pcts):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
            f'{pct:.0f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('TS_Results/fig12_optimal_rate.png')
plt.close()

# --- Runtime vs. cost scatter ---
fig, ax = plt.subplots(figsize=(9, 6))
for exp in experiments:
    sub = results_df[results_df['experiment'] == exp]
    ax.scatter(sub['runtime_s'], sub['best_cost'], c=COLORS[exp],
               label=EXP_LABELS_SHORT[exp], s=80, edgecolors='black', alpha=0.8, zorder=3)
ax.axhline(OPTIMAL_COST, color='red', linestyle='--', linewidth=1, alpha=0.6, label='Optimal')
ax.set_xlabel('Runtime (seconds)')
ax.set_ylabel('Best Cost Found')
#ax.set_title('Runtime vs. Best Cost Across All 35 Runs')  # caption in report
ax.legend(fontsize=8, loc='upper right')
plt.tight_layout()
plt.savefig('TS_Results/fig13_runtime_vs_cost.png')
plt.close()

# --- Mean ± Std bar chart ---
fig, ax = plt.subplots(figsize=(10, 5))
means = [results_df[results_df['experiment'] == e]['best_cost'].mean() for e in experiments]
stds = [results_df[results_df['experiment'] == e]['best_cost'].std() for e in experiments]
bars = ax.bar(range(len(experiments)), means, yerr=stds, capsize=6,
              color=[COLORS[e] for e in experiments], edgecolor='black', alpha=0.85)
ax.axhline(OPTIMAL_COST, color='red', linestyle='--', linewidth=1.5, label=f'Optimal ({OPTIMAL_COST})')
ax.set_xticks(range(len(experiments)))
ax.set_xticklabels([EXP_LABELS_SHORT[e] for e in experiments], rotation=20, ha='right')
ax.set_ylabel('Best Cost (Mean ± Std)')
#ax.set_title('Mean Best Cost with Standard Deviation by Experiment')  # caption in report
ax.legend()
y_min = max(OPTIMAL_COST - 5, min(m - s for m, s in zip(means, stds)) - 5)
y_max = max(m + s for m, s in zip(means, stds)) + 10
ax.set_ylim(y_min, y_max)
plt.tight_layout()
plt.savefig('TS_Results/fig14_mean_std_bar.png')
plt.close()

# --- Grid layout of best solution ---
best_perm_str = results_df.loc[results_df['best_cost'].idxmin(), 'best_perm'] if 'best_perm' in results_df.columns else None
if best_perm_str is None:
    full_df2 = pd.read_csv('TS_Results/all_35_runs_full.csv')
    best_perm_str = full_df2.loc[full_df2['best_cost'].idxmin(), 'best_perm_str']
    nums = [int(n) for n in re.findall(r'\d+', best_perm_str) if int(n) < 15]
else:
    nums = list(best_perm_str) if isinstance(best_perm_str, list) else [int(n) for n in re.findall(r'\d+', str(best_perm_str)) if int(n) < 15]

fig, ax = plt.subplots(figsize=(8, 5))
grid = np.zeros((3, 5), dtype=int)
for dept in range(15):
    loc = nums[dept]
    grid[loc // 5][loc % 5] = dept + 1
for r in range(3):
    for c in range(5):
        color = '#D5E8F0' if (r + c) % 2 == 0 else '#E8F4E8'
        rect = plt.Rectangle((c, 2 - r), 1, 1, facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(c + 0.5, 2 - r + 0.5, f'Dept {grid[r][c]}',
                ha='center', va='center', fontsize=12, fontweight='bold')
ax.set_xlim(0, 5); ax.set_ylim(0, 3); ax.set_aspect('equal')
ax.set_xticks([0.5, 1.5, 2.5, 3.5, 4.5])
ax.set_xticklabels(['Col 1', 'Col 2', 'Col 3', 'Col 4', 'Col 5'])
ax.set_yticks([0.5, 1.5, 2.5])
ax.set_yticklabels(['Row 3', 'Row 2', 'Row 1'])
#ax.set_title(f'Best Solution Layout (Cost = {int(results_df["best_cost"].min())})', fontsize=14)  # caption in report
plt.tight_layout()
plt.savefig('TS_Results/fig15_grid_layout.png')
plt.close()

# --- Statistical tests ---
print("\n" + "=" * 70)
print("STATISTICAL ANALYSIS")
print("=" * 70)
groups = [results_df[results_df['experiment'] == e]['best_cost'].values for e in experiments]
F_stat, p_val = f_oneway(*groups)
print(f"\nOne-Way ANOVA: F = {F_stat:.4f}, p = {p_val:.6f}")
comparisons = [
    ('E1_basic', 'E5_aspiration', 'Basic vs Aspiration'),
    ('E5_aspiration', 'E6_partial', 'Full NH vs Partial NH'),
    ('E5_aspiration', 'E7_diversification', 'No Diversif. vs Restart'),
    ('E2_small_tabu', 'E3_large_tabu', 'Small tabu vs Large tabu'),
]
print("\nPairwise t-tests:")
for e1, e2, label in comparisons:
    c1 = results_df[results_df['experiment'] == e1]['best_cost'].values
    c2 = results_df[results_df['experiment'] == e2]['best_cost'].values
    t_stat, p_val = ttest_ind(c1, c2)
    print(f"  {label}: t={t_stat:.3f}, p={p_val:.4f} {'*' if p_val < 0.05 else ''}")

print("\nAll enhanced plots saved to TS_Results/")
print("DONE.")
