import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def select_audio_features(df):
    """选择音频特征列"""
    audio_features = ['danceability', 'acousticness', 'energy',
                     'instrumentalness', 'liveness', 'loudness',
                     'speechiness', 'tempo']
    return df[audio_features].copy()

def preprocess_data(X):
    """数据预处理：处理缺失值"""
    X.fillna(X.mean(), inplace=True)
    return X

def standardize_features(X):
    """特征标准化"""
    scaler = StandardScaler()
    return scaler.fit_transform(X), scaler