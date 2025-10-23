# -*- coding: utf-8 -*-
"""
Collection of functions related to stripes detection

Created on Sun May 18 02:49:33 2025

@author: Vadym Sokhan
"""

# %%
import os
import glob

import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from scipy.stats import laplace

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


# %%

"""
------------------------------------
1. K-means clustering along PC1 
------------------------------------

"""

def kmeans_pc1(X, n_clusters=5):
    # Center the data
    X_centered = X - np.mean(X, axis=0)
    
    # Fit PCA and get PC1
    pca = PCA(n_components=1)
    pc1_projection = pca.fit_transform(X_centered)  # Shape: (n_samples, 1)
    
    # Run 1D k-means clustering on the PC1 projections
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(pc1_projection)
    
    return cluster_labels, pc1_projection.ravel(), pca.components_[0]

# %%

def kmeans_pc1_csv(csv_path, n_clusters=5, output_path=False, plot=True):
    """
    Loads X and Y columns from a CSV file, applies PCA,
    and clusters the data along the first principal component (PC1).
    Adds the cluster labels to the DataFrame and saves it to a new CSV
    with a '_kmean' suffix if output_path not specified.

    Parameters:
    - csv_path : str, path to the CSV file
    - n_clusters : int, number of clusters for KMeans
    - output_path : str or False, where to save the updated CSV
    - plot : bool, if True, saves a plot as PNG instead of displaying it

    Returns:
    - output_path : str, path to the saved CSV with cluster labels
    """
    # Load CSV
    df = pd.read_csv(csv_path)
    
    if not {"X", "Y"}.issubset(df.columns):
        raise ValueError("CSV must contain 'X' and 'Y' columns.")
    
    data = df[["X", "Y"]].values

    # Run PCA + KMeans
    cluster_labels, _, _ = kmeans_pc1(data, n_clusters=n_clusters)

    # Add labels (starting from 1)
    df["Stripe_Kmean"] = cluster_labels + 1

    # Determine CSV save path
    if output_path is False:
        base, ext = os.path.splitext(csv_path)
        output_path = base + ".csv"
    
    df.to_csv(output_path, index=False)
    print(f"Clustered data saved to: {output_path}")

    # Save plot
    if plot:
        folder = os.path.dirname(csv_path)
        base = os.path.splitext(os.path.basename(csv_path))[0]
        plot_path = os.path.join(folder, base + "_kmean_plot.png")
    
        plt.figure(figsize=(8, 8))
        plt.scatter(df["X"], df["Y"], c=cluster_labels, cmap='tab10', s=40, edgecolor='k')
    
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("KMeans Clustering Along PC1")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"Plot saved to: {plot_path}")

    return output_path

# %%

def all_kmeans_pc1(root_folder, n_clusters=5):
    """
    Recursively searches all subfolders of the given root folder,
    and applies kmeans_along_pc1_from_csv to the first CSV found in each.

    Parameters:
    - root_folder : str, path to the top-level directory
    - n_clusters : int, number of clusters for each CSV

    Returns:
    - results : list of paths to output CSV files
    """
    results = []

    for subdir, dirs, files in os.walk(root_folder):

        # Look for CSV files in the current subdirectory
        csv_files = [f for f in files if f.lower().endswith('_results.csv')]
        if not csv_files:
            continue  # skip if no CSVs

        # Use the first CSV file found in this subdirectory
        csv_path = os.path.join(subdir, csv_files[0])

        try:
            print(f"Processing: {csv_path}")
            output_path =kmeans_pc1_csv(csv_path, n_clusters=n_clusters)
            results.append(output_path)
        except Exception as e:
            print(f"Failed to process {csv_path}: {e}")

    print(f"\n✅ Processed {len(results)} file(s).")
    return results

 
# %%
"""
------------------------------------
2. Mixture of polynomial regressions
------------------------------------

"""

# Function to generate curved clusters
def generate_curved_clusters(n_points=100, n_clusters=5, noise=0.02, x_shift=.7):
    np.random.seed(42)
    X = []
    for i in range(n_clusters):
        radius = 1 + i * x_shift 
        t = np.linspace(0, 2 * np.pi, n_points)
        x_raw = radius * np.cos(t) + noise * np.random.randn(n_points) + i * x_shift
        y_raw = radius * np.sin(t) + noise * np.random.randn(n_points)
        
        
        
        mask = (y_raw > -1.1) & (y_raw < 1.1)  & (x_raw > 0) # Create a boolean mask
        x = x_raw[mask]
        y = y_raw[mask]
        
        X.append(np.column_stack([x, y]))
    return np.vstack(X)


# %%

def rotate_to_pc2(x, y):
    """
    Rotates (x, y) data so that the second principal component (PC2) aligns with the x-axis.

    Parameters:
        x (array-like): Array of x coordinates.
        y (array-like): Array of y coordinates.

    Returns:
        x_rot (np.ndarray): Rotated x coordinates (aligned with PC2).
        y_rot (np.ndarray): Rotated y coordinates.
    """
    # Convert inputs to numpy arrays and stack them
    X = np.column_stack((x, y))

    # Center the data
    X_centered = X - np.mean(X, axis=0)

    # Perform PCA
    pca = PCA(n_components=2)
    pca.fit(X_centered)

    # Create new axes with PC2 aligned to x-axis
    new_axes = np.array([pca.components_[1], pca.components_[0]])  # PC2, PC1

    # Rotate data
    X_rotated = X_centered @ new_axes.T

    # Split into x and y components
    x_rot, y_rot = X_rotated[:, 0], X_rotated[:, 1]

    return x_rot, y_rot

def rotate_points(x, y, angle_deg):
    angle_rad = np.deg2rad(angle_deg)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    x_rot = cos_a * x - sin_a * y
    y_rot = sin_a * x + cos_a * y
    return x_rot, y_rot

def compute_weighted_rss(X, y, betas, responsibilities):
    """
    Compute weighted residual sum of squares over all components:
    sum over k, i of R[i,k] * (y[i] - prediction_k(x[i]))^2
    """
    n_samples, _ = X.shape
    n_components = betas.shape[0]
    rss = 0.0
    for k in range(n_components):
        y_pred = X @ betas[k]
        residuals = y - y_pred
        rss += np.sum(responsibilities[:, k] * residuals**2)
    return rss


# %%

def initialize_responsibilities_by_y(x, y, n_components):
    y = np.asarray(y).reshape(-1)
    N = len(y)
    responsibilities = np.zeros((N, n_components))

    # Split y into bins using percentiles
    split = np.percentile(y, 100 * np.arange(1, n_components) / n_components)

    for i in range(N):
        for k in range(n_components):
            if k == 0 and y[i] <= split[0]:
                responsibilities[i, k] = 1
            elif k == n_components - 1 and y[i] > split[-1]:
                responsibilities[i, k] = 1
            elif 0 < k < n_components - 1 and split[k - 1] < y[i] <= split[k]:
                responsibilities[i, k] = 1

    responsibilities /= responsibilities.sum(axis=1, keepdims=True)
    return responsibilities


# %%

def fit_mix_polinom(x, y, degree=2, n_components=5, n_iters=50, random_state=42, 
                    gif_path=None):
    np.random.seed(random_state)
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)
    X = np.column_stack([x**i for i in range(degree + 1)])

    responsibilities = initialize_responsibilities_by_y(x, y, n_components)

    betas = np.random.randn(n_components, degree + 1)
    variances = np.ones(n_components)
    
    colors = ['red', 'green', 'blue', 'gray', 'yellow']
    cmap = ListedColormap(colors)
    
    if gif_path:
        frame_dir = os.path.join(os.path.dirname(gif_path), "em_frames")
        os.makedirs(frame_dir, exist_ok=True)
        frame_paths = []

    for iter_num in range(n_iters):
        # M-step WITHOUT regularization
        for k in range(n_components):
            r = responsibilities[:, k]
            W = np.diag(r)
            XtWX = X.T @ W @ X
            XtWy = X.T @ W @ y

            # Solve the weighted least squares problem without regularization
            betas[k] = np.linalg.solve(XtWX, XtWy)

            y_pred = X @ betas[k]
            variances[k] = np.sum(r * (y - y_pred) ** 2) / r.sum()

        # E-step (unchanged)
        for k in range(n_components):
            mu = X @ betas[k]
            var = variances[k]
            responsibilities[:, k] = (1 / np.sqrt(2 * np.pi * var)) * np.exp(-0.5 * (y - mu) ** 2 / var)
        responsibilities /= responsibilities.sum(axis=1, keepdims=True)

        predicted_labels = responsibilities.argmax(axis=1)

        # Plotting for GIF (unchanged)
        if gif_path:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(x, y, c=predicted_labels, cmap=cmap, s=20, alpha=0.6)
            for i, beta in enumerate(betas):
                x_i = x[predicted_labels == i]
                if len(x_i) == 0:
                    continue
                sort_idx = np.argsort(x_i)
                x_sorted = x_i[sort_idx]
                X_plot = np.column_stack([x_sorted**j for j in range(degree + 1)])
                y_fit = X_plot @ beta
                ax.plot(x_sorted, y_fit, color=colors[i], linewidth=2, label=f'Component {i+1}')
        
            ax.set_title(f'EM Iteration {iter_num+1}')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.legend()
            ax.grid(True)
            plt.tight_layout()
        
            frame_path = os.path.join(frame_dir, f"frame_{iter_num:03d}.png")
            plt.savefig(frame_path)
            frame_paths.append(frame_path)
            plt.close(fig)
    
    if gif_path:
        import imageio.v2 as imageio
        with imageio.get_writer(gif_path, mode='I', duration=0.4, loop=0) as writer:
            for frame_path in frame_paths:
                writer.append_data(imageio.imread(frame_path))
        print(f"GIF saved to {gif_path}")
        
    return betas, responsibilities, predicted_labels

# %%


def fit_mix_polinom_csv(
    csv_path,
    degree=2,
    n_components=5,
    n_iters=50,
    random_state=42,
    mult_init=False
):
    """
    Loads X,Y from CSV, fits mixture of polynomial regressions via EM,
    saves a plot instead of showing it, and writes results to CSV.

    Args:
        csv_path (str): Path to input CSV.
        degree (int): Polynomial degree.
        n_components (int): Number of mixture components.
        n_iters (int): Number of EM iterations.
        random_state (int): Random seed.
        mult_init (bool): Use multi-angle initialization.

    Returns:
        (betas, responsibilities, out_csv_path)
    """
    df = pd.read_csv(csv_path)
    if not {"X", "Y"}.issubset(df.columns):
        raise ValueError("CSV must contain 'X' and 'Y' columns")
    x = df["X"].values
    y = df["Y"].values

    x, y = rotate_to_pc2(x, y)

    if mult_init:
        best_rss = np.inf
        best_betas = best_resps = best_labels = None

        for rotation_idx in range(89):
            angle = 4 * rotation_idx
            x_rot, y_rot = rotate_points(x, y, angle)
            X = np.column_stack([x_rot**i for i in range(degree + 1)])

            betas, responsibilities, labels = fit_mix_polinom(
                x_rot, y_rot, degree=degree, n_components=n_components,
                n_iters=n_iters, random_state=random_state)

            rss = compute_weighted_rss(X, y_rot, betas, responsibilities)
            if rss < best_rss:
                best_rss = rss
                best_betas = betas
                best_resps = responsibilities
                best_labels = labels

        betas = best_betas
        responsibilities = best_resps
        labels = best_labels
    else:
        betas, responsibilities, labels = fit_mix_polinom(
            x, y, degree=degree, n_components=n_components,
            n_iters=n_iters, random_state=random_state)

    # Save plot as image
    colors = ['red', 'green', 'blue', 'gray', 'yellow']
    cmap = ListedColormap(colors[:n_components])

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(x, y, c=labels, cmap=cmap, s=20, alpha=0.6)
    for k in range(n_components):
        xs = np.sort(x)
        Xp = np.column_stack([xs**i for i in range(degree + 1)])
        ax.plot(xs, Xp @ betas[k], color=colors[k], linewidth=2, label=f'Comp {k+1}')
    ax.set_title('Final Mixture Fit')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()

    plot_path = os.path.splitext(csv_path)[0] + "_pol_plot.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved plot to: {plot_path}")

    # Save CSV with labels
    df["Stripe_Polreg"] = labels + 1
    out_csv = csv_path
    df.to_csv(out_csv, index=False)
    print(f"Saved CSV with labels: {out_csv}")

    return betas, responsibilities, out_csv

# %%

def all_mix_polinom(root_dir):
    """
    Traverse through subdirectories of root_dir, apply data rotation to each '_results.csv' file,
    save the rotated data as '_rot.csv', and then apply polynomial fitting to each '_rot.csv' file.
    """
    # Step 1: Locate all '_colonies.csv' files in subdirectories
    pattern = os.path.join(root_dir, '**', '*_results.csv')
    csv_files = glob.glob(pattern, recursive=True)

    for csv_path in csv_files:
        print(f"Processing file: {csv_path}")
        fit_mix_polinom_csv(csv_path, degree=1)

# %%

def summary_stripes(root_folder, cluster_column="Stripe", file_extension="_results.csv", n_clusters=5):
    """
    Scans all CSV files with a specific extension in a folder tree and counts the number of entries in each cluster.
    Saves the result to a summary CSV with the cluster_column name as a suffix.

    Parameters:
    - root_folder : str, root directory to search recursively
    - cluster_column : str, name of the column containing cluster labels
    - file_extension : str, file extension to filter files (default is '_kmean.csv')

    Returns:
    - output_path : str, path to the saved summary CSV
    """
    results = []

    for subdir, _, files in os.walk(root_folder):
        for file in files:
            if file.endswith(file_extension):
                full_path = os.path.join(subdir, file)
                sample = os.path.relpath(full_path, root_folder)
                sample = sample.split(os.sep)[0]

                try:
                    df = pd.read_csv(full_path)

                    if cluster_column not in df.columns:
                        print(f"⚠️ Column '{cluster_column}' not found in {file}, skipping.")
                        continue

                    # Count entries per cluster
                    counts = df[cluster_column].value_counts().to_dict()
                    for cid in range(1, n_clusters + 1):
                        results.append({
                            "file": sample,
                            "cluster": cid,
                            cluster_column: counts.get(cid, 0)
                        })

                except Exception as e:
                    print(f"❌ Failed to read or process {file}: {e}")

    # Create summary DataFrame
    summary_df = pd.DataFrame(results)

    # Build output path with suffix
    output_path = os.path.join(root_folder, f"stripes_summary.csv")
    if os.path.exists(output_path):
        old_df = pd.read_csv(output_path)
        summary_df = old_df.merge(summary_df, on = ['file', 'cluster'])
        
    summary_df.to_csv(output_path, index=False)
    print(f"\n✅ Summary saved to: {output_path}")

    return output_path

