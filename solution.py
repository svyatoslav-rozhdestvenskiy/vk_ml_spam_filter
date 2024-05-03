import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

df_train = pd.read_csv('datasets/train_spam.csv')
df_test = pd.read_csv('datasets/test_spam.csv')

print(df_train.head(10))
print(df_train.shape)
print(df_train.dtypes.value_counts())

print(df_test.head(10))
print(df_test.shape)
print(df_test.dtypes.value_counts())

print(df_train['text_type'].value_counts())


def len_distribution(df):
    text_len = df['text'].apply(len)
    print(f'Минимальная длина сообщения {text_len.min()}')
    print(f'Максимальная длина сообщения {text_len.max()}')
    print(f'Средняя длина сообщения {text_len.mean()}')
    print(f'Медиана длин сообщений {text_len.median()}')
    fig = plt.figure(figsize=(15, 15))
    bins = 30
    fig.add_subplot(1, 1, 1)
    plt.hist(text_len, bins=bins, color='skyblue', edgecolor='black')
    plt.axvline(text_len.mean(), color='red', linestyle='solid', linewidth=3)
    plt.text(text_len.mean() + 5, 700, f'Среднее', color='red')
    plt.axvline(text_len.median(), color='green', linestyle='solid', linewidth=3)
    plt.text(text_len.median() + 5, 700, f'Медиана', color='green')
    step = (text_len.max() - text_len.min()) / bins
    ticks = [text_len.min() + i * step for i in range(bins + 1)]
    plt.xticks(ticks, rotation=90)
    plt.title('Распределение длины текстов')
    plt.xlabel('Длина текста')
    plt.ylabel('Количество сообщений')
    plt.grid(True)
    plt.show()


print('\nРаспределение тренировочной выборки\n')
len_distribution(df_train)
print('\nРаспределение тестовой выборки\n')
len_distribution(df_test)
df_spam = df_train[df_train['text_type'] == 'spam']
df_ham = df_train[df_train['text_type'] == 'ham']
print('\nРаспределение не спам\n')
len_distribution(df_ham)
print('\nРаспределение спам\n')
len_distribution(df_spam)
df_short_spam = df_spam[df_spam['text'].apply(len) <= 20]
df_short_ham = df_ham[df_ham['text'].apply(len) <= 10]
print(df_short_spam.head(20))
print(df_short_ham.head(20))
x_train, x_test, y_train, y_test = train_test_split(df_train['text'], df_train['text_type'], shuffle=True, test_size=0.2, random_state=42)
y_test = y_test.apply(lambda x: 1 if x == 'spam' else 0)
y_train = y_train.apply(lambda x: 1 if x == 'spam' else 0)
print(x_train.head())
print(x_train.shape)
print(x_test.head())
print(x_test.shape)
print(y_train.head())
print(y_train.shape)
print(y_test.head())
print(y_test.shape)
count_vec = CountVectorizer(stop_words='english')
count_vec.fit(x_train)
x_train_count_vec = count_vec.transform(x_train)
x_test_count_vec = count_vec.transform(x_test)
tfidf_vec = TfidfVectorizer(stop_words='english')
tfidf_vec.fit(x_train)
x_train_tfidf_vec = tfidf_vec.transform(x_train)
x_test_tfidf_vec = tfidf_vec.transform(x_test)


def calculate_roc_auc(y_predict, y_test=y_test, model_name='модель'):
    print(f'\nROC-AUC для модели: {model_name}\n')
    roc_auc = roc_auc_score(y_test, y_predict)
    print('ROC AUC=%.3f' % (roc_auc))
    fpr, tpr, thresholds = roc_curve(y_test, y_predict)
    plt.plot(fpr, tpr, color='darkorange',
             label='ROC кривая (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name}')
    plt.legend(loc="lower right")
    plt.show()


logreg_model_count = LogisticRegression(random_state=42)
logreg_model_count.fit(x_train_count_vec, y_train)
calculate_roc_auc(logreg_model_count.predict_proba(x_test_count_vec)[:, 1],
                  model_name='Логистическая регрессия с count векторизацией')
logreg_model_tfidf = LogisticRegression(random_state=42)
logreg_model_tfidf.fit(x_train_tfidf_vec, y_train)
calculate_roc_auc(logreg_model_tfidf.predict_proba(x_test_tfidf_vec)[:, 1],
                  model_name='Логистическая регрессия с TF-IDF векторизацией')

rf_model_count = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model_count.fit(x_train_count_vec, y_train)
calculate_roc_auc(rf_model_count.predict_proba(x_test_count_vec)[:, 1],
                  model_name='Случайный лес с count векторизацией')
rf_model_tfidf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model_tfidf.fit(x_train_tfidf_vec, y_train)
calculate_roc_auc(rf_model_tfidf.predict_proba(x_test_tfidf_vec)[:, 1],
                  model_name='Случайный лес с TF-IDF векторизацией')

xgb_model_count = XGBClassifier(n_estimators=100, max_depth=6,
                                use_label_encoder=False, eval_metric='auc', random_state=42)
xgb_model_count.fit(x_train_count_vec, y_train)
calculate_roc_auc(xgb_model_count.predict_proba(x_test_count_vec)[:, 1],
                  model_name='Градиентный бустинг с count векторизацией')
xgb_model_tfidf = XGBClassifier(n_estimators=100, max_depth=6,
                                use_label_encoder=False, eval_metric='auc', random_state=42)
xgb_model_tfidf.fit(x_train_tfidf_vec, y_train)
calculate_roc_auc(xgb_model_tfidf.predict_proba(x_test_tfidf_vec)[:, 1],
                  model_name='Градиентный бустинг с TF-IDF векторизацией')
test_vec = count_vec.transform(df_test['text'])
predict_test = logreg_model_count.predict_proba(test_vec)[:, 1]
df_test['score'] = pd.DataFrame(predict_test)
df_test.to_csv('datasets/test_score.csv')