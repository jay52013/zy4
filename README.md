# 数据分析
- name，album，artist，release_date，这些非数值数据对我后续的模型没有帮助，所以删去。
- danceability, acousticness, energy, instrumentalness, liveness, loudness, speechiness, tempode，这些是音乐的各类指标，都可以在之后的分析里起到作用，所以要留下。
# 特征处理
- 对数值特征进行标准化处理。
# 模型处理
- 自动选择簇数K​
​肘部法​：绘制SSE随K变化的曲线，选择“肘点”（下降速度骤减的K值）。
​轮廓系数​：衡量簇内紧密度和簇间分离度，选择系数最大的K值。
