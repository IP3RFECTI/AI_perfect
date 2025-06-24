# dataset_pipeline.py — универсальный шаблон с полными логами обработки датасета
import plotly.express as px
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_selection import SelectKBest, f_classif
from statsmodels.stats.outliers_influence import variance_inflation_factor
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
from sklearn.base import clone
from imblearn.over_sampling import SMOTE
from collections import Counter
import os
import warnings
warnings.filterwarnings('ignore')

# === Загрузка ===
filename = "iris.json"  # Заменить на ваш файл
df = pd.read_json(filename)
print("\n✅ Датасет загружен. Размер:", df.shape)

# === Анализ признаков ===
def analyze_feature_types(df):
    print("\n🔍 Типы признаков:")
    for col in df.columns:
        if df[col].dtype == 'object' or df[col].nunique() < 20:
            print(f"- {col}: категориальный ({df[col].nunique()} уникальных)")
        else:
            print(f"- {col}: числовой")

analyze_feature_types(df)

# === Визуализация пропусков ===
print("\n📊 Визуализация пропусков (missingno):")
msno.matrix(df)
plt.title("Пропущенные значения")
plt.show()

# === Удаление дубликатов ===
duplicates = df.duplicated().sum()
df = df.drop_duplicates()
print(f"\n✅ Удалено дубликатов: {duplicates}. Новый размер: {df.shape}")

# === Обработка пропусков ===
print("\n🔁 Обработка пропусков:")
for col in df.columns:
    null_count = df[col].isnull().sum()
    if null_count == 0:
        continue
    if df[col].dtype == 'object':
        fill_value = df[col].mode()[0]
        df[col].fillna(fill_value, inplace=True)
        print(f"- {col}: заменено {null_count} → мода '{fill_value}'")
    else:
        fill_value = df[col].median()
        df[col].fillna(fill_value, inplace=True)
        print(f"- {col}: заменено {null_count} → медиана {fill_value}")

# === Удаление постоянных признаков ===
constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
df.drop(columns=constant_cols, inplace=True)
print(f"\n⚠️ Удалено постоянных признаков: {constant_cols if constant_cols else 'нет'}")

# === Обнаружение и удаление выбросов ===
print("\n🚨 Обнаружение выбросов (IsolationForest):")
numeric_cols = df.select_dtypes(include=[np.number]).columns
iso = IsolationForest(contamination=0.03, random_state=42)
outliers = iso.fit_predict(df[numeric_cols])
outlier_count = (outliers == -1).sum()
df = df[outliers == 1]
print(f"Удалено выбросов: {outlier_count}. Размер после очистки: {df.shape}")

# === Графики распределения признаков ===
print("\n📊 Графики распределения признаков (гистограммы и боксплоты):")
import math

numeric_cols = df.select_dtypes(include=[np.number]).columns
n_cols = len(numeric_cols)
n_rows = math.ceil(n_cols / 2)

plt.figure(figsize=(15, n_rows * 4))
for i, col in enumerate(numeric_cols):
    plt.subplot(n_rows, 2, i + 1)
    sns.histplot(df[col], kde=True, bins=30, color='skyblue')
    plt.title(f'Распределение: {col}')
plt.tight_layout()
plt.show()

plt.figure(figsize=(15, n_rows * 4))
for i, col in enumerate(numeric_cols):
    plt.subplot(n_rows, 2, i + 1)
    sns.boxplot(x=df[col], color='lightcoral')
    plt.title(f'Боксплот: {col}')
plt.tight_layout()
plt.show()

for i in df.columns:
    fig = px.histogram(df,
                   x=i,
                   marginal='box',
                   text_auto=True,
                   color_discrete_sequence  = ['forestgreen'],
                   template='simple_white',
                   title=i.upper() + ' HISTOGRAM')

    fig.update_layout(xaxis_title=i,yaxis_title="Count", bargap=0.1)

    fig.show()

# === Кодирование категориальных признаков ===
def encode_categorical_features(df):
    df_encoded = df.copy()
    label_encoders = {}
    print("\n🧠 Кодирование категориальных признаков:")
    for col in df.columns:
        if df[col].dtype == 'object' or df[col].nunique() < 20:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df[col])
            label_encoders[col] = le
            print(f"- {col}: закодирован (LabelEncoder) → {dict(zip(le.classes_, le.transform(le.classes_)))})")
    return df_encoded, label_encoders

df, label_encoders = encode_categorical_features(df)

# === Сохранение обработанного датасета ===
processed_name = os.path.splitext(filename)[0] + "_обработанный.csv"
df.to_csv(processed_name, index=False)
print(f"\n💾 Обработанный датасет сохранён: {processed_name}")

# === Мультиколлинеарность (VIF) ===
def compute_vif(df_num):
    print("\n📉 Расчёт мультиколлинеарности (VIF):")
    from statsmodels.tools.tools import add_constant
    X_const = add_constant(df_num)
    vif = pd.DataFrame()
    vif["Feature"] = df_num.columns
    vif["VIF"] = [variance_inflation_factor(X_const.values, i + 1) for i in range(df_num.shape[1])]
    print(vif)

compute_vif(df[numeric_cols])

# === Целевая переменная ===
target_col = 'species'  # ← Замените на свою
X = df.drop(columns=[target_col])
y = df[target_col]

# === Масштабирование ===
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
print("\n📏 Признаки масштабированы (StandardScaler)")

# === SelectKBest ===
selector = SelectKBest(score_func=f_classif, k=10)
selector.fit(X_scaled, y)
top_features = X_scaled.columns[selector.get_support()]
X_selected = X_scaled[top_features]
print(f"\n✅ SelectKBest выбрал топ-10 признаков: {list(top_features)}")

# === Exhaustive Feature Search ===
print("\n🔍 Exhaustive Feature Search (mlxtend):")
base_model = LogisticRegression(max_iter=1000)
efs = EFS(clone(base_model),
          min_features=2,
          max_features=4,
          scoring='accuracy',
          print_progress=True,
          cv=3)
efs = efs.fit(X_selected, y)
X_selected = X_selected[list(efs.best_feature_names_)]
print(f"✅ Лучшая комбинация признаков (EFS): {list(efs.best_feature_names_)}")

# === SMOTE балансировка ===
X_bal, y_bal = SMOTE(random_state=42).fit_resample(X_selected, y)
print(f"\n⚖️ Балансировка классов (SMOTE): до {Counter(y)}, после {Counter(y_bal)}")

# === Разделение выборки ===
X_train, X_temp, y_train, y_temp = train_test_split(X_bal, y_bal, test_size=0.3, stratify=y_bal, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)
print(f"\n📂 Разделение: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")

# === Обучение моделей ===
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier()
}

for name, model in models.items():
    print(f"\n🚀 Обучение модели: {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    print(f"🔍 Результаты ({name}):")
    print(classification_report(y_val, y_pred))

# === Важность признаков (Random Forest) ===
rf = models['Random Forest']
importances = rf.feature_importances_
sorted_idx = np.argsort(importances)[::-1]
top_feat = X_selected.columns[sorted_idx]

plt.figure(figsize=(10, 5))
sns.barplot(x=importances[sorted_idx], y=top_feat, palette="Blues_d")
plt.title("🌟 Важность признаков (Random Forest)")
plt.tight_layout()
plt.show()

# === Финальный отчёт ===
print("\n📌 Итог:")
print("""
✔️ Дубликаты, пропуски, выбросы устранены
✔️ Категориальные признаки закодированы
✔️ Выполнена визуализация пропусков
✔️ Применён VIF-анализ мультиколлинеарности
✔️ Отбор признаков: SelectKBest + Exhaustive Search
✔️ Балансировка классов через SMOTE
✔️ Обучены модели: LogisticRegression, RandomForest, GradientBoosting
✔️ Построена визуализация важности признаков
""")