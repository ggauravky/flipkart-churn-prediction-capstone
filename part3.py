"""Part 3: Segment customers with K-means, compare hierarchical linkages, and run PCA.

Writes outputs/part3_results.txt, part3_segments.csv and part3_hierarchical.csv.
"""

import csv
import os
import numpy as np
import data_loader


def format_value(val):
    """Format a result as the brief requires: yes/no, plain integer, or 4-decimal float."""
    # bool must be checked before int, because bool is a subclass of int.
    if isinstance(val, (bool, np.bool_)):
        return "yes" if val else "no"
    elif isinstance(val, (int, np.integer)):
        return str(val)
    elif isinstance(val, (float, np.floating)):
        formatted = f"{val:.4f}"
        return "0.0000" if formatted == "-0.0000" else formatted
    return str(val)


def run_kmeans(X_std, K):
    """Run K-means with the brief's pinned rules and return (wcss, labels, centroids, passes).

    - Start from rows 0, floor(n/K), 2*floor(n/K), ... so every run is reproducible.
    - Assign each row to its nearest centroid by squared Euclidean distance.
    - Move each centroid to its members' mean; an empty cluster keeps its centroid.
    - Stop at the first pass in which no row changes cluster (no iteration cap).
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
        # argmin returns the first minimum, so ties go to the lower-numbered centroid.
        new_assignments = np.argmin(dists, axis=1)

        if np.array_equal(new_assignments, assignments):
            break

        assignments = new_assignments

        for k in range(K):
            members = X_std[assignments == k]
            if len(members) > 0:
                centroids[k] = np.mean(members, axis=0)

    # Within-cluster sum of squares in standardized space
    wcss = float(np.sum((X_std - centroids[assignments]) ** 2))
    return wcss, assignments, centroids, passes


def run_hierarchical_clustering(dist_matrix, labels):
    """Agglomerative clustering with single, complete and average linkage.

    Linkage distance between two clusters is the min (single), max (complete)
    or mean (average) of all member-to-member distances. Returns every merge
    and, per linkage, the number of clusters left at the dendrogram cut.
    """
    all_merges = []
    clusters_at_cut = {}

    for linkage in ["single", "complete", "average"]:
        clusters = [[i] for i in range(len(labels))]
        merges = []

        for merge_no in range(1, len(labels)):
            min_dist = float("inf")
            best_pair = None

            # Find the closest pair of clusters under this linkage rule.
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

            # The brief pins cluster_a as the cluster with the smaller lowest member.
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

            new_clusters = [clusters[k] for k in range(len(clusters)) if k != i and k != j]
            new_clusters.append(sorted(c1 + c2))
            clusters = new_clusters

        # Cut the dendrogram in the largest gap between consecutive merge heights.
        # gaps[k] lies between merge k+1 and merge k+2, so k+1 merges sit below the cut.
        heights = [m["height"] for m in merges]
        gaps = [heights[k + 1] - heights[k] for k in range(len(heights) - 1)]
        max_gap_idx = int(np.argmax(gaps))
        merges_below_cut = max_gap_idx + 1
        num_clusters = len(labels) - merges_below_cut
        clusters_at_cut[linkage] = num_clusters

        all_merges.extend(merges)

    return all_merges, clusters_at_cut


def run_part3():
    dataset = data_loader.load_dataset()
    X_orig = dataset["X"][:, :5]  # continuous predictors X1..X5
    y = dataset["y"]
    val = dataset["value"]
    n_samples = len(X_orig)

    # Part 3 is unsupervised, so all 800 rows are standardised together (sample SD).
    mean_X = np.mean(X_orig, axis=0)
    sd_X = np.std(X_orig, axis=0, ddof=1)
    X_std = (X_orig - mean_X) / sd_X

    results = {}

    # Task 3.1 — K-means from scratch for K = 2, 3, 4, 5, 6
    wcss_results = {}
    assignments_k3 = None

    for K in [2, 3, 4, 5, 6]:
        wcss, assign, _, _ = run_kmeans(X_std, K)
        wcss_results[K] = wcss
        results[f"wcss_k{K}"] = wcss
        if K == 3:
            assignments_k3 = assign

    # Task 3.2 — Choose K with the elbow
    # Smallest K in {3, 4, 5} such that WCSS(K) - WCSS(K+1) < 0.5 * (WCSS(K-1) - WCSS(K))
    elbow_k = None
    for K_cand in [3, 4, 5]:
        drop_curr = wcss_results[K_cand - 1] - wcss_results[K_cand]
        drop_next = wcss_results[K_cand] - wcss_results[K_cand + 1]
        if drop_next < 0.5 * drop_curr:
            elbow_k = K_cand
            break

    results["elbow_k"] = elbow_k

    # Task 3.3 — Profile the K = 3 segments
    # Cluster numbers are arbitrary, so label them A, B, C by ascending mean X3
    # (discount share, original units) to get a stable naming.
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

    # Task 3.4 — Hierarchical clustering on the supplied table
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

    # Task 3.5 — PCA by the seven steps
    # Step 1: mean vector - already zero (to floating-point precision) after standardising.
    # Step 2: centre anyway, so the steps match the brief exactly.
    X_c = X_std - np.mean(X_std, axis=0)

    # Step 3: covariance matrix S = (1 / (n - 1)) X^T X
    S = (1.0 / (n_samples - 1)) * (X_c.T @ X_c)

    # Step 4: eigh is meant for symmetric matrices like S (the one library routine allowed).
    eigenvalues, eigenvectors = np.linalg.eigh(S)

    # Step 5: eigh returns ascending order, so reverse to put the largest first.
    sort_idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sort_idx]
    eigenvectors = eigenvectors[:, sort_idx]

    # An eigenvector's sign is arbitrary. The brief pins it: flip each one so that
    # its largest-magnitude entry is positive, making loadings reproducible.
    for j in range(eigenvectors.shape[1]):
        max_idx = np.argmax(np.abs(eigenvectors[:, j]))
        if eigenvectors[max_idx, j] < 0:
            eigenvectors[:, j] = -eigenvectors[:, j]

    # Step 6: keep the top two eigenvectors
    W = eigenvectors[:, :2]

    # Step 7: project Z = X_c W
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

    # Write outputs/part3_results.txt
    results_path = os.path.join("outputs", "part3_results.txt")
    with open(results_path, "w", newline="\n", encoding="utf-8") as f:
        for key, value in results.items():
            f.write(f"{key} = {format_value(value)}\n")

    print(f"Generated {results_path} with {len(results)} keys.")
    return results


if __name__ == "__main__":
    run_part3()

