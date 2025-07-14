import pandas as pd


def analyze_clusters(df, audio_features):
    """分析聚类结果"""
    # 查看每个聚类的歌曲数量
    print("Cluster distribution:")
    print(df['cluster'].value_counts())

    # 查看每个聚类的特征均值
    cluster_means = df.groupby('cluster')[audio_features].mean()
    print("\nCluster characteristics:")
    print(cluster_means)

    return cluster_means