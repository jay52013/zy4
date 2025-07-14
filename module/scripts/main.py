import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from feature_processing import select_audio_features, preprocess_data, standardize_features
from clustering import find_optimal_k, perform_clustering
from visualization import plot_elbow_and_silhouette, plot_radar_chart
from data_analysis import analyze_clusters


def main():
    # 1. 读取数据
    data_path = r'E:\PycharmProjects\pythonProject4\module\data\nigerian-songs.csv'

    # 检查文件是否存在
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"数据文件未找到，请检查路径: {data_path}")

    df = pd.read_csv(data_path)

    # 2. 选择音频特征列
    audio_features = ['danceability', 'acousticness', 'energy',
                      'instrumentalness', 'liveness', 'loudness',
                      'speechiness', 'tempo']
    X = select_audio_features(df)

    # 3. 数据预处理
    X = preprocess_data(X)

    # 4. 特征标准化
    X_scaled, scaler = standardize_features(X)

    # 5. 使用肘部法则确定最佳K值
    k_range = range(2, 11)
    inertia, silhouette_scores = find_optimal_k(X_scaled, k_range)

    # 可视化肘部法则和轮廓系数
    plot_elbow_and_silhouette(k_range, inertia, silhouette_scores)

    # 6. 选择最佳K值（这里示例选择3，实际应根据图表选择）
    best_k = 3  # 应该根据可视化结果选择最佳K值

    # 7. 使用最佳K值进行聚类
    kmeans = perform_clustering(X_scaled, best_k)

    # 8. 将聚类结果添加到原始数据
    df['cluster'] = kmeans.labels_

    # 9. 分析聚类结果
    cluster_means = analyze_clusters(df, audio_features)

    # 10. 可视化聚类特征（雷达图）
    plot_radar_chart(cluster_means, audio_features)


if __name__ == "__main__":
    main()