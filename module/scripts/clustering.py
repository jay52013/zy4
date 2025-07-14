import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def find_optimal_k(X_scaled, k_range=range(2, 11)):
    """使用肘部法则和轮廓系数确定最佳K值"""
    inertia = []
    silhouette_scores = []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertia.append(kmeans.inertia_)

        # 计算轮廓系数（数据量大时可能较慢）
        if len(X_scaled) > 10000:  # 如果数据量太大可跳过
            silhouette_scores.append(None)
        else:
            score = silhouette_score(X_scaled, kmeans.labels_)
            silhouette_scores.append(score)

    return inertia, silhouette_scores


def perform_clustering(X_scaled, best_k):
    """使用最佳K值进行聚类"""
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    return kmeans