"""Part 3: Segmentation and dimensionality reduction.

Flipkart E-commerce Decision Intelligence Capstone.
This script performs:
1. K-means clustering from scratch on features X1..X5 across all 800 rows for K in {2, 3, 4, 5, 6}.
2. Pinned starting row selection, tie-breaking, and WCSS computation.
3. Elbow K determination via the official drop threshold formula.
4. Profiling the three segments (A, B, C ordered by ascending mean of X3) -> outputs/part3_segments.csv.
5. Agglomerative hierarchical clustering from scratch (single, complete, average) on the 6-cohort table -> outputs/part3_hierarchical.csv.
6. Principal Component Analysis (PCA) via the seven-step manual linear algebra procedure.

Outputs written to:
- outputs/part3_results.txt
- outputs/part3_segments.csv
- outputs/part3_hierarchical.csv
"""

import csv
import math
import os
import numpy as np
import data_loader


def format_value(val):
    """Format a value according to the capstone specification:
    - booleans as yes / no (checked before int since bool subclasses int)
    - integers as plain integers
    - 4 decimal places for floating-point numbers
    - strings as given
    """
    if isinstance(val, (bool, np.bool_)):
        return "yes" if val else "no"
    elif isinstance(val, (int, np.integer)):
        return str(val)
    elif isinstance(val, (float, np.floating)):
        formatted = f"{val:.4f}"
        return "0.0000" if formatted == "-0.0000" else formatted
    return str(val)


def run_kmeans(X_std, K):
    """Run K-means with exact pinned rules:
    - Starting centroids: rows at indices 0, floor(800/K), 2*floor(800/K), ...
    - Squared Euclidean distance assignment with lower-index tie-breaking.
    - Centroids updated to mean of cluster members (empty clusters unchanged).
    - Stops at first pass where no row changes cluster.
    """
    n_samples = len(X_std)
    start_indices = [j * (n_samples // K) for j in range(K)]
    centroids = X_std[start_indices].copy()
    assignments = np.full(n_samples, -1)

    passes = 0
    while True:
        passes += 1
        # dists shape: (n_samples, K)
        dists = np.sum((X_std[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2, axis=2)
        new_assignments = np.argmin(dists, axis=1)

        # Convergence: stop when no row changes assignment
        if np.array_equal(new_assignments, assignments):
            break

        assignments = new_assignments

        # Update centroids: mean of assigned members
        for k in range(K):
            members = X_std[assignments == k]
            if len(members) > 0:
                centroids[k] = np.mean(members, axis=0)

    # Within-cluster sum of squares in standardized space
    wcss = float(np.sum((X_std - centroids[assignments]) ** 2))
    return wcss, assignments, centroids, passes


def run_hierarchical_clustering(dist_matrix, labels):
    """Agglomerative clustering from scratch for single, complete, and average linkage.

    Merges clusters with smallest linkage distance at each step.
    Computes cut position at the largest consecutive gap in merge heights.
    """
    all_merges = []
    clusters_at_cut = {}

    for linkage in ["single", "complete", "average"]:
        clusters = [[i] for i in range(len(labels))]
        merges = []

        for merge_no in range(1, len(labels)):
            min_dist = float("inf")
            best_pair = None

            # Find closest pair of clusters under current linkage definition
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    c1, c2 = clusters[i], clusters[j]
                    pair_dists = [dist_matrix[m1, m2] for m1 in c1 for m2 in c2]

                    if linkage == "single":
                        d = min(pair_dists)
                    elif linkage == "complete":
                        d = max(pair_dists)
                    elif linkage == "average":
                        d = sum(pair_dists) / len(pair_dists)

                    if d < min_dist:
                        min_dist = d
                        best_pair = (i, j)

            i, j = best_pair
            c1, c2 = clusters[i], clusters[j]

            # cluster_a is cluster with the smaller minimum member index
            if min(c1) < min(c2):
                ca, cb = c1, c2
            else:
                ca, cb = c2, c1

            name_a = "+".join(sorted([labels[m] for m in ca]))
            name_b = "+".join(sorted([labels[m] for m in cb]))

            merges.append({
                "linkage": linkage,
                "merge_no": merge_no,
                "cluster_a": name_a,
                "cluster_b": name_b,
                "height": min_dist,
                "new_cluster": sorted(c1 + c2)
            })

            # Form new cluster list
            new_clusters = [clusters[k] for k in range(len(clusters)) if k != i and k != j]
            new_clusters.append(sorted(c1 + c2))
            clusters = new_clusters

        # Dendrogram cut: largest gap among consecutive merge heights
        heights = [m["height"] for m in merges]
        gaps = [heights[k + 1] - heights[k] for k in range(len(heights) - 1)]
        max_gap_idx = int(np.argmax(gaps))  # gap between merge (idx+1) and merge (idx+2)
        # Number of clusters is 6 minus number of merges below the cut
        merges_below_cut = max_gap_idx + 1
        num_clusters = len(labels) - merges_below_cut
        clusters_at_cut[linkage] = num_clusters

        all_merges.extend(merges)

    return all_merges, clusters_at_cut


def run_part3():
    # Load dataset across all 800 rows
    dataset = data_loader.load_dataset()
    X_orig = dataset["X"][:, :5]  # Continuous predictors X1..X5
    y = dataset["y"]              # Outcome (churned)
    val = dataset["value"]        # Order value in hundreds of INR
    n_samples = len(X_orig)

    # Standardise X1..X5 over all 800 rows using sample SD (ddof=1)
    mean_X = np.mean(X_orig, axis=0)
    sd_X = np.std(X_orig, axis=0, ddof=1)
    X_std = (X_orig - mean_X) / sd_X

    results = {}

    # =========================================================================
    # Task 3.1: K-means from scratch for K = 2, 3, 4, 5, 6
    # =========================================================================
    wcss_results = {}
    assignments_k3 = None

    for K in [2, 3, 4, 5, 6]:
        wcss, assign, _, _ = run_kmeans(X_std, K)
        wcss_results[K] = wcss
        results[f"wcss_k{K}"] = wcss
        if K == 3:
            assignments_k3 = assign

    # =========================================================================
    # Task 3.2: Choose K with the elbow
    # =========================================================================
    # Smallest K in {3, 4, 5} such that WCSS(K) - WCSS(K+1) < 0.5 * (WCSS(K-1) - WCSS(K))
    elbow_k = None
    for K_cand in [3, 4, 5]:
        drop_curr = wcss_results[K_cand - 1] - wcss_results[K_cand]
        drop_next = wcss_results[K_cand] - wcss_results[K_cand + 1]
        if drop_next < 0.5 * drop_curr:
            elbow_k = K_cand
            break

    results["elbow_k"] = elbow_k

    # =========================================================================
    # Task 3.3: Profile the K = 3 segments
    # =========================================================================
    # Order the three clusters by ascending mean of original X3 (discount_order_share)
    cluster_x3_means = []
    for k in range(3):
        mask = (assignments_k3 == k)
        cluster_x3_means.append((k, float(np.mean(X_orig[mask, 2]))))

    cluster_x3_means.sort(key=lambda item: item[1])
    segment_labels = ["A", "B", "C"]

    segments_rows = []
    segment_masks = {}

    for seg_label, (k_orig, _) in zip(segment_labels, cluster_x3_means):
        mask = (assignments_k3 == k_orig)
        segment_masks[seg_label] = mask
        size = int(np.sum(mask))
        results[f"size_segment_{seg_label}"] = size

        m_x1 = float(np.mean(X_orig[mask, 0]))
        m_x2 = float(np.mean(X_orig[mask, 1]))
        m_x3 = float(np.mean(X_orig[mask, 2]))
        m_x4 = float(np.mean(X_orig[mask, 3]))
        m_x5 = float(np.mean(X_orig[mask, 4]))
        outcome_rate = float(np.mean(y[mask]))
        mean_val = float(np.mean(val[mask]))

        segments_rows.append([
            seg_label,
            size,
            f"{m_x1:.4f}",
            f"{m_x2:.4f}",
            f"{m_x3:.4f}",
            f"{m_x4:.4f}",
            f"{m_x5:.4f}",
            f"{outcome_rate:.4f}",
            f"{mean_val:.4f}"
        ])

    os.makedirs("outputs", exist_ok=True)
    segments_csv_path = os.path.join("outputs", "part3_segments.csv")
    with open(segments_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "segment", "size", "mean_X1", "mean_X2", "mean_X3", "mean_X4", "mean_X5", "outcome_rate", "mean_value"
        ])
        writer.writerows(segments_rows)
    print(f"Generated {segments_csv_path} with {len(segments_rows)} rows.")

    # =========================================================================
    # Task 3.4: Hierarchical clustering on the supplied table
    # =========================================================================
    cohort_labels = ["E1", "E2", "E3", "E4", "E5", "E6"]
    dist_table = np.array([
        [0.0, 7.9, 7.4, 2.9, 2.9, 6.1],
        [7.9, 0.0, 1.4, 5.9, 5.3, 2.0],
        [7.4, 1.4, 0.0, 5.8, 4.6, 1.3],
        [2.9, 5.9, 5.8, 0.0, 2.8, 4.5],
        [2.9, 5.3, 4.6, 2.8, 0.0, 3.4],
        [6.1, 2.0, 1.3, 4.5, 3.4, 0.0]
    ], dtype=np.float64)

    hier_merges, cuts = run_hierarchical_clustering(dist_table, cohort_labels)

    hierarchical_csv_path = os.path.join("outputs", "part3_hierarchical.csv")
    with open(hierarchical_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["linkage", "merge_no", "cluster_a", "cluster_b", "height"])
        for m in hier_merges:
            writer.writerow([
                m["linkage"],
                m["merge_no"],
                m["cluster_a"],
                m["cluster_b"],
                f"{m['height']:.4f}"
            ])
    print(f"Generated {hierarchical_csv_path} with {len(hier_merges)} rows.")

    results["single_clusters_at_cut"] = cuts["single"]
    results["complete_clusters_at_cut"] = cuts["complete"]
    results["average_clusters_at_cut"] = cuts["average"]

    # =========================================================================
    # Task 3.5: PCA by the seven steps
    # =========================================================================
    # Step 1: Mean vector (all zeros after standardising)
    # Step 2: Centre the data
    X_c = X_std - np.mean(X_std, axis=0)

    # Step 3: Covariance matrix S = (1 / (n - 1)) X^T X
    S = (1.0 / (n_samples - 1)) * (X_c.T @ X_c)

    # Step 4: Eigenvalues and eigenvectors via numpy.linalg.eigh
    eigenvalues, eigenvectors = np.linalg.eigh(S)

    # Step 5: Sort by eigenvalue, largest first
    sort_idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sort_idx]
    eigenvectors = eigenvectors[:, sort_idx]

    # Sign convention: flip each eigenvector so entry of largest absolute value is positive
    for j in range(eigenvectors.shape[1]):
        max_idx = np.argmax(np.abs(eigenvectors[:, j]))
        if eigenvectors[max_idx, j] < 0:
            eigenvectors[:, j] = -eigenvectors[:, j]

    # Step 6: Keep top two eigenvectors
    W = eigenvectors[:, :2]

    # Step 7: Project Z = X_c @ W
    Z = X_c @ W

    eigenvalue_sum = float(np.sum(eigenvalues))
    results["eigenvalue_sum"] = eigenvalue_sum

    for i in range(5):
        results[f"eigenvalue_{i+1}"] = float(eigenvalues[i])

    explained_ratios = eigenvalues / eigenvalue_sum
    for i in range(5):
        results[f"explained_ratio_{i+1}"] = float(explained_ratios[i])

    cum_ratios = np.cumsum(explained_ratios)
    for i in range(5):
        results[f"cumulative_ratio_{i+1}"] = float(cum_ratios[i])

    components_80 = int(np.where(cum_ratios >= 0.8)[0][0] + 1)
    results["components_for_80pct"] = components_80

    # Loadings of top 2 components (W columns)
    for i in range(5):
        results[f"pc1_loading_X{i+1}"] = float(W[i, 0])
    for i in range(5):
        results[f"pc2_loading_X{i+1}"] = float(W[i, 1])

    # Projections of first three data rows (indices 0, 1, 2)
    results["row1_pc1"] = float(Z[0, 0])
    results["row1_pc2"] = float(Z[0, 1])
    results["row2_pc1"] = float(Z[1, 0])
    results["row2_pc2"] = float(Z[1, 1])
    results["row3_pc1"] = float(Z[2, 0])
    results["row3_pc2"] = float(Z[2, 1])

    # Mean PC1 score per segment A, B, C
    pc1_scores = Z[:, 0]
    for seg_label in ["A", "B", "C"]:
        mask = segment_masks[seg_label]
        results[f"mean_pc1_segment_{seg_label}"] = float(np.mean(pc1_scores[mask]))

    # =========================================================================
    # Write outputs/part3_results.txt
    # =========================================================================
    results_path = os.path.join("outputs", "part3_results.txt")
    with open(results_path, "w", newline="\n", encoding="utf-8") as f:
        for key, val in results.items():
            f.write(f"{key} = {format_value(val)}\n")

    print(f"Generated {results_path} with {len(results)} keys.")
    return results


if __name__ == "__main__":
    run_part3()

