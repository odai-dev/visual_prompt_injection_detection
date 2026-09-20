"""
Unsupervised Learning Module.

Applies K-Means clustering (k=3) on the TF-IDF feature space to discover
latent semantic attack groupings, then projects clusters into 2D via PCA
for interpretable visualisation.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")  # non-interactive backend safe for headless environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"

KMEANS_PATH = MODELS_DIR / "kmeans_model.joblib"
PCA_PLOT_PATH = FIGURES_DIR / "kmeans_clusters_pca.png"

# Cluster theme labels (annotated by inspecting top-term profiles)
CLUSTER_LABELS = {
    0: "Cluster 0 – Benign Notices & Signage",
    1: "Cluster 1 – System Prompt Exfiltration",
    2: "Cluster 2 – Jailbreak & Command Override",
}

CLUSTER_COLORS = ["#2ecc71", "#e74c3c", "#e67e22"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def train_kmeans(
    train_path: Path = PROCESSED_DIR / "train.csv",
    vectorizer_path: Path = MODELS_DIR / "tfidf_vectorizer.joblib",
    n_clusters: int = 3,
    random_state: int = 42,
    verbose: bool = True,
) -> Tuple[KMeans, np.ndarray, List[str]]:
    """
    Fit K-Means on the TF-IDF feature space of the training corpus.

    Parameters
    ----------
    train_path : Path
        Training split CSV.
    vectorizer_path : Path
        Path to the persisted TF-IDF vectorizer.
    n_clusters : int
        Number of clusters (default 3).
    random_state : int
        Reproducibility seed.

    Returns
    -------
    Tuple[KMeans, np.ndarray, List[str]]
        (fitted_kmeans, cluster_assignments, feature_names)
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(train_path)
    text_col = "cleaned_text" if "cleaned_text" in train_df.columns else "text"
    texts = train_df[text_col].fillna("[EMPTY_IMAGE]").tolist()
    labels = train_df["label"].tolist() if "label" in train_df.columns else None

    vectorizer = joblib.load(vectorizer_path)
    X = vectorizer.transform(texts)

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_assignments = km.fit_predict(X)

    if verbose:
        for c in range(n_clusters):
            count = int((cluster_assignments == c).sum())
            logger.info("Cluster %d: %d samples.", c, count)

    # Persist model
    joblib.dump(km, KMEANS_PATH)
    if verbose:
        logger.info("K-Means model saved → %s", KMEANS_PATH.name)

    feature_names = vectorizer.get_feature_names_out().tolist()

    # Generate PCA visualisation
    _plot_kmeans_pca(X, cluster_assignments, labels, n_clusters, verbose=verbose)

    return km, cluster_assignments, feature_names


def get_top_cluster_terms(
    km: KMeans,
    vectorizer,
    n_terms: int = 10,
) -> Dict[int, List[str]]:
    """
    Return top TF-IDF terms per cluster center.

    Parameters
    ----------
    km : KMeans
        Fitted K-Means model.
    vectorizer : TfidfVectorizer
        Fitted TF-IDF vectorizer.
    n_terms : int
        Number of top terms to return per cluster.

    Returns
    -------
    Dict[int, List[str]]
        Mapping of cluster index → top term strings.
    """
    terms = vectorizer.get_feature_names_out()
    top_terms = {}
    for idx, center in enumerate(km.cluster_centers_):
        top_indices = center.argsort()[::-1][:n_terms]
        top_terms[idx] = [terms[i] for i in top_indices]
    return top_terms


def load_kmeans(model_path: Path = KMEANS_PATH) -> KMeans:
    """Load a persisted K-Means model from disk."""
    return joblib.load(model_path)


def train_unsupervised_clustering(
    train_csv: Path,
    vectorizer_path: Path,
    models_dir: Path,
    figures_dir: Path,
    n_clusters: int = 3,
    verbose: bool = False,
) -> Dict:
    """
    Alias for :func:`train_kmeans` using positional-style args expected by tests.

    Returns a dict with 'model_path', 'plot_path', and 'n_clusters' keys.
    """
    global KMEANS_PATH, PCA_PLOT_PATH, MODELS_DIR, FIGURES_DIR
    # Temporarily remap output paths if custom dirs provided
    orig_km = KMEANS_PATH
    orig_pca = PCA_PLOT_PATH

    KMEANS_PATH = models_dir / "kmeans_model.joblib"
    PCA_PLOT_PATH = figures_dir / "kmeans_clusters_pca.png"
    models_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    try:
        km, assignments, feature_names = train_kmeans(
            train_path=train_csv,
            vectorizer_path=vectorizer_path,
            n_clusters=n_clusters,
            verbose=verbose,
        )
    finally:
        KMEANS_PATH = orig_km
        PCA_PLOT_PATH = orig_pca

    return {
        "model_path": str(models_dir / "kmeans_model.joblib"),
        "plot_path": str(figures_dir / "kmeans_clusters_pca.png"),
        "n_clusters": n_clusters,
        "cluster_assignments": assignments.tolist(),
    }


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def _plot_kmeans_pca(
    X,
    cluster_assignments: np.ndarray,
    true_labels: Optional[List[int]],
    n_clusters: int,
    verbose: bool = True,
):
    """
    Project TF-IDF vectors to 2D via PCA and produce a coloured scatter plot.
    """
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X.toarray())

    fig, ax = plt.subplots(figsize=(10, 7))
    for c in range(n_clusters):
        mask = cluster_assignments == c
        label = CLUSTER_LABELS.get(c, f"Cluster {c}")
        color = CLUSTER_COLORS[c % len(CLUSTER_COLORS)]
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            c=color,
            label=label,
            alpha=0.65,
            edgecolors="white",
            linewidths=0.4,
            s=40,
        )

    explained = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC₁ ({explained[0]*100:.1f}% variance)", fontsize=12)
    ax.set_ylabel(f"PC₂ ({explained[1]*100:.1f}% variance)", fontsize=12)
    ax.set_title("K-Means Clustering (k=3) — 2D PCA Projection\nVisual Prompt Injection Detection", fontsize=14)
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(PCA_PLOT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)

    if verbose:
        logger.info("PCA cluster plot saved → %s", PCA_PLOT_PATH.name)
