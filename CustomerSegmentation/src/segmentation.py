"""Сегментація клієнтів Online Retail за RFM-показниками та K-means."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ["Recency", "Frequency", "Monetary"]


def load_data():
    return pd.read_excel(ROOT / "data" / "Online Retail.xlsx")


def clean_transactions(data):
    data = data.copy()
    data["InvoiceNo"] = data["InvoiceNo"].astype("string").str.strip()
    data["InvoiceDate"] = pd.to_datetime(data["InvoiceDate"], errors="coerce")
    for column in ["Quantity", "UnitPrice"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=["CustomerID", "InvoiceNo", "InvoiceDate", "Quantity", "UnitPrice"])
    cancelled = data["InvoiceNo"].str.upper().str.startswith("C")
    data = data.loc[~cancelled & (data["Quantity"] > 0) & (data["UnitPrice"] > 0)].copy()
    data["CustomerID"] = data["CustomerID"].astype(int)
    data["TotalPrice"] = data["Quantity"] * data["UnitPrice"]
    return data


def calculate_rfm(data):
    snapshot = data["InvoiceDate"].max().normalize() + pd.DateOffset(days=1)
    rfm = data.groupby("CustomerID").agg(
        LastPurchase=("InvoiceDate", "max"),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("TotalPrice", "sum"),
    )
    rfm["Recency"] = (snapshot - rfm["LastPurchase"].dt.normalize()).dt.days
    return rfm[FEATURES]


def prepare_features(rfm):
    # Логарифмування послаблює вплив великих покупок; стандартизація вирівнює масштаби.
    transformed = np.log1p(rfm[FEATURES].to_numpy(dtype=float))
    return StandardScaler().fit_transform(transformed)


def pick_k(x):
    rows = []
    for k in range(2, 9):
        model = KMeans(n_clusters=k, random_state=42, n_init=10).fit(x)
        silhouette = silhouette_score(x, model.labels_, sample_size=min(2500, len(x)), random_state=42)
        rows.append({"k": k, "inertia": model.inertia_, "silhouette": silhouette})
    scores = pd.DataFrame(rows)
    best_k = int(scores.loc[scores["silhouette"].idxmax(), "k"])
    return best_k, scores


def profile_clusters(rfm):
    profiles = rfm.groupby("Cluster").agg(
        Customers=("Recency", "size"),
        Recency_median=("Recency", "median"),
        Frequency_median=("Frequency", "median"),
        Monetary_median=("Monetary", "median"),
        Monetary_total=("Monetary", "sum"),
    ).round(2)
    profiles["Share_percent"] = (100 * profiles["Customers"] / len(rfm)).round(2)
    profiles["Revenue_share_percent"] = (100 * profiles["Monetary_total"] / rfm["Monetary"].sum()).round(2)
    return profiles


def charts(rfm, x, scores, profiles, output):
    plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False})

    # Розподіли RFM-показників.
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, column, color in zip(axes, FEATURES, ["#2B6777", "#52AB98", "#D38A53"]):
        ax.hist(np.log1p(rfm[column]), bins=40, color=color, edgecolor="white", alpha=.9)
        ax.set(xlabel=f"log(1 + {column})", ylabel="Customers", title=column)
    fig.suptitle("RFM feature distributions (log scale)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(output / "01_rfm_distributions.png")
    plt.close(fig)

    # Зв'язки між показниками.
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    pairs = [("Recency", "Frequency"), ("Recency", "Monetary"), ("Frequency", "Monetary")]
    for ax, (a, b) in zip(axes, pairs):
        ax.scatter(np.log1p(rfm[a]), np.log1p(rfm[b]), s=8, alpha=.3, color="#2B6777")
        ax.set(xlabel=f"log(1 + {a})", ylabel=f"log(1 + {b})", title=f"{a} vs {b}")
    fig.suptitle("Relationships between RFM features", fontweight="bold")
    fig.tight_layout()
    fig.savefig(output / "02_rfm_relationships.png")
    plt.close(fig)

    # Вибір кількості кластерів.
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax2 = ax1.twinx()
    ax1.plot(scores["k"], scores["inertia"], "o-", color="#2B6777")
    ax2.plot(scores["k"], scores["silhouette"], "s--", color="#DB8A48")
    ax1.set(xlabel="Number of clusters (k)", ylabel="Inertia", title="K-means: Elbow and Silhouette")
    ax2.set_ylabel("Silhouette coefficient")
    fig.tight_layout()
    fig.savefig(output / "03_k_selection.png")
    plt.close(fig)

    # PCA відображає готові кластери на площині.
    pca = PCA(n_components=2)
    projected = pca.fit_transform(x)
    fig, ax = plt.subplots(figsize=(8, 5))
    for cluster in sorted(rfm["Cluster"].unique()):
        mask = rfm["Cluster"].to_numpy() == cluster
        ax.scatter(projected[mask, 0], projected[mask, 1], s=8, alpha=.4, label=f"Cluster {cluster}")
    ax.legend(markerscale=2)
    ax.set(
        xlabel=f"PC1 ({pca.explained_variance_ratio_[0]:.1%})",
        ylabel=f"PC2 ({pca.explained_variance_ratio_[1]:.1%})",
        title="Clusters projected with PCA",
    )
    fig.tight_layout()
    fig.savefig(output / "04_pca_clusters.png")
    plt.close(fig)

    # Кількість клієнтів і частка виручки.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    profiles["Customers"].plot.bar(ax=axes[0], color="#2B6777", rot=0, title="Customer count by cluster", ylabel="Customers")
    profiles["Revenue_share_percent"].plot.bar(ax=axes[1], color="#DB8A48", rot=0, title="Revenue by cluster", ylabel="Revenue share, %")
    fig.tight_layout()
    fig.savefig(output / "05_cluster_sizes_revenue.png")
    plt.close(fig)

    # Теплова карта медіан кластерів.
    medians = rfm.groupby("Cluster")[FEATURES].median()
    normalized = np.log1p(medians)
    normalized = (normalized - normalized.mean()) / normalized.std(ddof=0).replace(0, 1)
    fig, ax = plt.subplots(figsize=(7, 4))
    image = ax.imshow(normalized, aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2)
    ax.set_xticks(range(3), FEATURES)
    ax.set_yticks(range(len(medians)), [f"Cluster {cluster}" for cluster in medians.index])
    ax.set_title("Cluster median profiles (standardized log scale)")
    fig.colorbar(image, ax=ax, label="Relative to clusters")
    fig.tight_layout()
    fig.savefig(output / "06_cluster_heatmap.png")
    plt.close(fig)


def main():
    output = ROOT / "results"
    output.mkdir(exist_ok=True)
    data = load_data()
    clean = clean_transactions(data)
    rfm = calculate_rfm(clean)
    x = prepare_features(rfm)
    best_k, scores = pick_k(x)

    model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    rfm["Cluster"] = model.fit_predict(x)
    profiles = profile_clusters(rfm)

    rfm.to_csv(output / "customer_segments.csv", encoding="utf-8-sig")
    profiles.to_csv(output / "cluster_profiles.csv", encoding="utf-8-sig")
    scores.to_csv(output / "k_evaluation.csv", index=False)
    charts(rfm, x, scores, profiles, output)

    print(f"Rows: {len(data)} -> {len(clean)}; customers: {len(rfm)}")
    print(f"Selected k: {best_k}; Silhouette: {scores.loc[scores['k'] == best_k, 'silhouette'].iloc[0]:.4f}")
    print(profiles.to_string())


if __name__ == "__main__":
    main()
