import pandas as pd
import numpy as np
import string
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import joblib

# 读取数据
data = pd.read_csv('IMDB_Dataset_SPANISH.csv')

# 数据预处理
def clean_text(text):
    # 去除标点符号
    text = text.translate(str.maketrans('', '', string.punctuation))
    # 转换为小写
    text = text.lower()
    # 移除停用词
    stop_words = stopwords.words('spanish')
    text = ' '.join([word for word in text.split() if word not in stop_words])
    # 词干提取
    stemmer = SnowballStemmer('spanish')
    text = ' '.join([stemmer.stem(word) for word in text.split()])
    return text


# 数据预处理
data['content'] = data['content'].apply(lambda x: clean_text(x))

# 划分数据集
train_size = int(len(data) * 0.8)
train_data = data[:train_size]
test_data = data[train_size:]

# 特征工程
vectorizer = CountVectorizer()
train_features = vectorizer.fit_transform(train_data['content'])
test_features = vectorizer.transform(test_data['content'])
joblib.dump(vectorizer, 'Model/count_vectorizer.pkl')
# 构建模型
clf = MultinomialNB()

# 模型训练
clf.fit(train_features, train_data['label'])

print("-------------------Train Finished!-------------------")
joblib.dump(clf, 'Model/NaiveBayes_model.pkl')
# 测试集预测
pred = clf.predict(test_features)
pred_prob = clf.predict_proba(test_features)

# 计算准确率
acc = accuracy_score(test_data['label'], pred)
print("Test Set Accuracy: ", acc)

# 测试样例
test_sample = ['Ella es mi mejor amiga,  me encanta mucho. ']
test_sample_vec = vectorizer.transform(test_sample)
test_pred = clf.predict(test_sample_vec)
test_prob = clf.predict_proba(test_sample_vec)
print('Test prediction:', test_pred)
print('Test probability:', test_prob)


# 加载模型
loaded_model = joblib.load('NaiveBayes_model.pkl')
# 直接使用加载的模型进行预测
test_sample = ['Ella es mi mejor amiga,  me encanta mucho. ']
test_sample_vec = vectorizer.transform(test_sample)
loaded_pred = loaded_model.predict(test_sample_vec)
loaded_prob = loaded_model.predict_proba(test_sample_vec)
print('Loaded prediction:', loaded_pred)
print('Loaded probability:', loaded_prob)
