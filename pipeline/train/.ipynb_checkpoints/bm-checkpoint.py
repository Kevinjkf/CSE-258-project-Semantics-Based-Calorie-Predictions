import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from tqdm import tqdm

from transformers import BertTokenizer, BertModel
import torch
from torch.utils.data import DataLoader, TensorDataset

def preprocess_text(text):
    words = word_tokenize(text)
    words = [word.lower() for word in words]
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word.isalnum() and word not in stop_words]
    return words

def build_vocabulary(corpus):
    vectorizer = CountVectorizer(tokenizer=preprocess_text, stop_words='english')
    vectorizer.fit(tqdm(corpus, desc='Building vocabulary'))
    return vectorizer

def compute_bow(corpus, vectorizer):
    return vectorizer.transform(tqdm(corpus, desc='Computing BoW'))

def compute_tfidf(corpus, word2ind):
    vectorizer = TfidfVectorizer(vocabulary=word2ind, tokenizer=preprocess_text, stop_words='english')
    return vectorizer.fit_transform(tqdm(corpus, desc='Computing TF-IDF'))

train_file = '../../dataset/processed/train_data.csv'
data = pd.read_csv(train_file)
steps_text = data.steps.tolist()

print("Preprocessing text and building vocabulary...")
vectorizer = build_vocabulary(steps_text)

print("Computing BoW...")
bow_matrix = compute_bow(steps_text, vectorizer)
print("BoW representation computed.")

print("Computing TF-IDF...")
tfidf_matrix = compute_tfidf(steps_text, vectorizer.vocabulary_)
print("TF-IDF matrix computed.")

test_file = '../../dataset/processed/train_proced.csv'
test_data = pd.read_csv(test_file)

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# 导入数据和BERT特征
valid_data = pd.read_csv(test_file)

# 定义BERT特征的维度和数据点的数量
n_samples = valid_data.shape[0]
n_features = 64

# 生成随机特征
random_features = np.random.rand(n_samples, n_features)

X_random = pd.DataFrame(random_features, columns=[f'random_feature_{i+1}' for i in range(n_features)])

# 准备训练数据（BERT特征列）
# X = valid_data[['bert_feature_{}'.format(i) for i in range(1,65)] + [ 'total_fat', 'sugar', 'sodium', 'protein', 'saturated_fat']]

# X_other_features = valid_data[['total_fat', 'sugar', 'sodium', 'protein', 'saturated_fat']]

# 合并随机特征和其他特征列
#X = pd.concat([pd.DataFrame(bert_embeddings)], axis=1)

#X = pd.concat([pd.DataFrame(tfidf_matrix)], axis=1)

X = pd.concat([pd.DataFrame(bow_matrix.toarray())], axis=1)

# 准备目标值（calories列）
y = valid_data['calories_log']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 初始化线性回归模型
regressor = LinearRegression()

# 拟合线性回归模型
regressor.fit(X_train, y_train)

# 使用模型对测试集进行预测
y_pred = regressor.predict(X_test)

# 计算预测的均方误差（MSE）
mse = mean_squared_error(y_test, y_pred)
print(f"The mean squared error (MSE) on test set: {mse:.2f}")

# 如果需要，可以保存模型，以便以后使用
# from joblib import dump
# dump(regressor, 'calories_regressor.joblib')
