import pandas as pd
import numpy as np
import random

def nearest_neighbor_tour(D, start):
    n = D.shape[0]
    unvisited = set(range(n)) - {start}
    tour = [start]
    current = start
    while unvisited:
        nxt = min(unvisited, key=lambda j: D[current, j])
        tour.append(nxt)
        unvisited.remove(nxt)
        current = nxt
    return tour

def two_opt(D, tour):
    n = len(tour)
    improved = True
    while improved:
        improved = False
        for i in range(1, n-2):
            for j in range(i+1, n):
                if j - i == 1:
                    continue
                a, b = tour[i-1], tour[i]
                c, d = tour[j-1], tour[j % n]
                delta = (D[a, c] + D[b, d]) - (D[a, b] + D[c, d])
                if delta < -1e-8:
                    tour[i:j] = reversed(tour[i:j])
                    improved = True
    return tour

def compute_tour_length(D, tour):
    return sum(D[tour[i], tour[(i+1)%len(tour)]] for i in range(len(tour)))

def best_tsp_tour(D, n_restarts=10):
    n = D.shape[0]
    best_tour, best_len = None, float('inf')
    for _ in range(n_restarts):
        start = random.randrange(n)
        tour = nearest_neighbor_tour(D, start)
        tour = two_opt(D, tour)
        L = compute_tour_length(D, tour)
        if L < best_len:
            best_len, best_tour = L, tour.copy()

    return best_tour

def cut_tour_at_max_gap(D, tour):
    n = len(tour)
    edge_dists = [D[tour[i], tour[(i+1)%n]] for i in range(n)]
    cut_idx = int(np.argmax(edge_dists))
    path = tour[cut_idx+1:] + tour[:cut_idx+1]
    return path

if __name__ == "__main__":
    INPUT_CSV  = "../2-kmeans/kmeans_centers_32.csv"
    OUTPUT_CSV = "kmeans_centers_32_ordered.csv"
    df = pd.read_csv(INPUT_CSV)
    X = df.values
    n = X.shape[0]
    D = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    best_cycle  = best_tsp_tour(D, n_restarts=20)
    linear_order = cut_tour_at_max_gap(D, best_cycle)
    mapping = {old: new for new, old in enumerate(linear_order)}
    df["new_index"] = df.index.map(mapping)
    df.sort_values("new_index", inplace=True)
    df.drop(columns=["new_index"], inplace=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print("✅ order codebooks saved:", OUTPUT_CSV)
