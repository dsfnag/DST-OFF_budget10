import numpy as np 
import pandas as pd 
pd.options.display.max_colwidth = 300

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor 
from sklearn import metrics

import re

RANDOM_SEED = 42
# !pip freeze > requirements.txt

import streamlit as st
import pandas as pd
import numpy as np
import plotly_express as px


st.write("""Загрузка исходных данных производится из файлов source_data.xls и titles.xls.
В файле source_data.xls введены и могут быть добавлены параметры прогноза социально-экономического развития, статьи доходов и расходов прошлых лет.
В файле titles.xls приведена расшифровка названий столбцов source_data.xls""")

df_input = st.cache(pd.read_excel)('source_data.xls')
df_headers = pd.read_excel('titles.xls')

st.write("\nЗагрузка данных произведена успешно\n")

# st.write("Парамерты СЭР",df_headers.head(47).set_index('пп'))
# st.write("Статьи бюджета", df_headers.tail(28).set_index('пп'))

type_multi = st.sidebar.multiselect('Выберите варианты прогноза СЭР для обучения модели', df_input['Тип'].unique(), 'базовый')
years_multi = st.sidebar.multiselect('Выберите годы для обучения модели', df_input['Период'].unique(), df_input[df_input['Период']<=2016]['Период'].unique())
year_target = st.sidebar.multiselect('Выберите год для тестирования или проверки', df_input['Период'].unique(), 2017)
sel_target = st.sidebar.multiselect('Выберите прогнозируемую статью баланса', (df_headers['пп'].astype(str)+'. '+df_headers['Наименование']).str.slice(0, 50).to_list(), "100. Доходы факт итого (без раздела 200), млн руб."[:50])
sel_pres = st.sidebar.multiselect('Выводимые данные', ['Корреляционный анализ', 'Прогноз', 'Значимые признаки'], ['Корреляционный анализ', 'Прогноз', 'Значимые признаки'])

df = df_input[df_input['Тип'].isin(type_multi)][['Период']+list(df_input.columns)[3:]]
df = df.set_axis(['Период']+(df_headers['пп'].astype(str)+'. '+df_headers['Наименование']).str.slice(0, 50).to_list(), axis='columns')
st.dataframe(df[df['Период'].isin(years_multi)])

df = df.copy().reset_index(drop=True).fillna(0)
key_param = sel_target[0]
st.write('Анализ статьи:', key_param)

# Для понимания значимости зависимости сгенерируем три случайные параметра
df['random'] = np.random.rand(len(df))
df['random_10'] = np.random.rand(len(df))*10000
df['random_sec'] = list(range(len(df)))
df['random_sec'] = df['random_sec'].sample(frac=1).reset_index(drop=True)

plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True
plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False

if 'Корреляционный анализ' in sel_pres:
    st.write("Корреляционный анализ 1/3")
    df_corr = df[[key_param]+[c for c in df.columns[:len(df.columns)//3] if not c in [key_param, 'Период']]].corr()
    # plt.figure(figsize=(16,10), dpi= 100)
    fig, ax = plt.subplots()
    fig.set_size_inches(16, 10)
    sns.heatmap(df_corr.abs(), annot=True, fmt='.2f', annot_kws={'fontsize':10}, cmap='Reds', linewidths=1)
    plt.xticks(horizontalalignment='left', fontweight='light', fontsize=10, rotation=30)
    plt.yticks(fontweight='light', fontsize=10);
    st.write(fig)
    
    st.write("Корреляционный анализ 2/3")
    
    df_corr = df[[key_param]+[c for c in df.columns[len(df.columns)//3:len(df.columns)//3*2] if not c in [key_param, 'Период']]].corr()
    # plt.figure(figsize=(16,10), dpi= 100)
    fig, ax = plt.subplots()
    fig.set_size_inches(16, 10)
    sns.heatmap(df_corr.abs(), annot=True, fmt='.2f', annot_kws={'fontsize':10}, cmap='Reds', linewidths=1)
    plt.xticks(horizontalalignment='left', fontweight='light', fontsize=10, rotation=30)
    plt.yticks(fontweight='light', fontsize=10);
    st.write(fig)
    
    st.write("Корреляционный анализ 3/3")
    df_corr = df[[key_param]+[c for c in df.columns[len(df.columns)//3*2:] if not c in [key_param, 'Период']]].corr()
    # plt.figure(figsize=(16,10), dpi= 100)
    fig, ax = plt.subplots()
    fig.set_size_inches(16, 10)
    sns.heatmap(df_corr.abs(), annot=True, fmt='.2f', annot_kws={'fontsize':10}, cmap='Reds', linewidths=1)
    plt.xticks(horizontalalignment='left', fontweight='light', fontsize=10, rotation=30)
    plt.yticks(fontweight='light', fontsize=10);
    st.write(fig)

# Создаём модель, используя модель Random forest
model = RandomForestRegressor(n_estimators=25, random_state=RANDOM_SEED)
df_train = df[df['Период'].isin(years_multi)]
y_train = df_train[key_param].values
X_train = df_train.drop('Период', axis=1).iloc[:,:47]

# обучим модель
model.fit(X_train, y_train)

df_test = df[df['Период'].isin(year_target)]

if year_target[0] not in [2021, 2022]:
    y_test = df_test[key_param].values

X_test = df_test.drop('Период', axis=1).iloc[:,:47]

y_pred = model.predict(X_test)
mean = df[key_param].mean()
st.write('Прогноз статьи '+key_param+' :', mean)

# Проверка при прогнозе до 2021 года
st.write('Год прогноза ', year_target[0])
if year_target[0] not in [2021, 2022]:
    mae = metrics.mean_absolute_error(y_test,y_pred)
    mae_percent = mae/mean
    st.write('средняя ошибка прогноза: ', mae)
    st.write('средняя ошибка в %:', mae_percent*100, '%')

if 'Значимые признаки' in sel_pres:
    # выведем самые важные признаки
    fig, ax = plt.subplots()
    fig.set_size_inches(12, 10)
    feat_importances = pd.Series(model.feature_importances_, index=X_test.columns)
    feat_importances.nsmallest(60).plot(kind='barh')
    st.write(fig)
