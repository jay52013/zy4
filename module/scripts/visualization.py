import matplotlib.pyplot as plt
import numpy as np


def plot_elbow_and_silhouette(k_range, inertia, silhouette_scores):
    """可视化肘部法则和轮廓系数"""
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(k_range, inertia, 'bo-')
    plt.xlabel('Number of clusters (K)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method')

    plt.subplot(1, 2, 2)
    plt.plot(k_range, silhouette_scores, 'go-')
    plt.xlabel('Number of clusters (K)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score')

    plt.tight_layout()
    plt.show()


def plot_radar_chart(cluster_means, features):
    """可视化聚类特征（雷达图）"""
    angles = np.linspace(0, 2 * np.pi, len(features), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))

    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))

    for idx, row in cluster_means.iterrows():
        values = row.values.flatten().tolist()
        values += values[:1]  # 闭合图形
        ax.plot(angles, values, 'o-', linewidth=2, label=f'Cluster {idx}')
        ax.fill(angles, values, alpha=0.25)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(features)
    ax.set_title('Audio Features by Cluster', size=20, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.show()