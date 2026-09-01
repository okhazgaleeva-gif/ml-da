# Лабораторная работа 1
## Многослойный перцептрон для предсказания риска ишемической болезни сердца

**Дисциплина:** Машинное обучение и анализ данных  
**Семестр:** 2  
**Инструменты:** Python 3, NumPy, pandas, scikit-learn, Matplotlib  
**Среда:** Google Colab (рекомендуется)

---

## 1. Теоретические сведения

### 1.1. Зачем нужны нейронные сети

В первом семестре вы изучали **линейные модели** — логистическую регрессию, линейную регрессию, метод опорных векторов. Эти модели строят **линейную** границу между классами (или линейную зависимость для регрессии). На практике зависимости между признаками и целевой переменной часто **нелинейны**.

**Пример из медицины:** риск ишемической болезни сердца (ИБС) может резко возрастать только при одновременном выполнении нескольких условий (высокий холестерин *и* низкая толерантность к нагрузке *и* определённый тип боли в груди). Линейная модель «усредняет» такие эффекты и часто работает хуже, чем модель, способная выявлять **комбинации признаков**.

**Нейронная сеть** — это композиция простых нелинейных преобразований, которая может аппроксимировать сложные зависимости.

---

### 1.2. Искусственный нейрон

**Искусственный нейрон** (модель McCulloch–Pitts, 1943) — элементарная вычислительная единица:

1. На вход поступает вектор признаков $\mathbf{x} = (x_1, x_2, \ldots, x_n)$.
2. К каждому входу применяется вес $w_i$ и добавляется смещение (bias) $b$.
3. Вычисляется **взвешенная сумма** (pre-activation):

$$
z = \sum_{i=1}^{n} w_i x_i + b = \mathbf{w}^T \mathbf{x} + b
$$

4. К $z$ применяется **функция активации** $f$:

$$
a = f(z)
$$

**Связь с логистической регрессией:** один нейрон с сигмоидой $\sigma(z) = \frac{1}{1 + e^{-z}}$ и функцией потерь cross-entropy — это **логистическая регрессия**. Нейронная сеть — обобщение: несколько нейронов, организованных в слои.

---

### 1.3. Функции активации

| Функция | Формула | Диапазон | Применение |
|---------|---------|----------|------------|
| **Sigmoid** | $\sigma(z) = \frac{1}{1+e^{-z}}$ | $(0, 1)$ | Выходной слой бинарной классификации |
| **ReLU** | $\text{ReLU}(z) = \max(0, z)$ | $[0, +\infty)$ | Скрытые слои (наиболее популярна) |
| **Softmax** | $\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}$ | $(0, 1)$, сумма = 1 | Выходной слой многоклассовой классификации |
| **Tanh** | $\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$ | $(-1, 1)$ | Скрытые слои (альтернатива sigmoid) |

**Производная ReLU:**

$$
\frac{d\,\text{ReLU}}{dz} = \begin{cases} 1, & z > 0 \\ 0, & z \leq 0 \end{cases}
$$

**Производная sigmoid:**

$$
\sigma'(z) = \sigma(z)\,(1 - \sigma(z))
$$

ReLU почти всегда предпочтительнее sigmoid в скрытых слоях: она проще вычисляется и уменьшает проблему **затухающих градиентов** (vanishing gradients).

---

### 1.4. Многослойный перцептрон (MLP)

**MLP (Multilayer Perceptron)** — сеть из последовательно соединённых слоёв:

```
Входной слой → Скрытый слой(и) → Выходной слой
```

- **Входной слой** — это сами признаки (не содержит обучаемых параметров).
- **Скрытый слой** — набор нейронов с нелинейной активацией (обычно ReLU).
- **Выходной слой** — один или несколько нейронов в зависимости от задачи.

**Теорема об универсальной аппроксимации:** MLP с одним скрытым слоем и достаточным числом нейронов может аппроксимировать любую непрерывную функцию с заданной точностью. На практике используют 1–3 скрытых слоя.

**Прямой проход (forward pass)** для сети с одним скрытым слоем:

$$
\mathbf{h} = \text{ReLU}(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1)
$$

$$
\hat{\mathbf{y}} = \text{softmax}(\mathbf{W}_2 \mathbf{h} + \mathbf{b}_2)
$$

где $\mathbf{W}_1, \mathbf{W}_2$ — матрицы весов, $\mathbf{b}_1, \mathbf{b}_2$ — векторы смещений.

---

### 1.5. Функция потерь

Для **многоклассовой классификации** (например, «здоров» / «болен» — два класса, но формула обобщается) используют **категориальную cross-entropy**:

$$
\mathcal{L} = -\frac{1}{m} \sum_{i=1}^{m} \sum_{c=1}^{C} y_{i,c} \log(\hat{y}_{i,c})
$$

где:

- $m$ — число объектов в батче;
- $C$ — число классов;
- $y_{i,c}$ — one-hot метка (1 для истинного класса, 0 для остальных);
- $\hat{y}_{i,c}$ — предсказанная вероятность класса $c$ для объекта $i$.

Для **бинарной классификации** (2 класса) cross-entropy упрощается до:

$$
\mathcal{L} = -\frac{1}{m} \sum_{i=1}^{m} \left[ y_i \log \hat{y}_i + (1 - y_i) \log (1 - \hat{y}_i) \right]
$$

**Связь с первым семестром:** cross-entropy для бинарного случая — та же функция потерь, что в логистической регрессии.

---

### 1.6. Алгоритм обратного распространения ошибки (Backpropagation)

**Backpropagation** — алгоритм вычисления градиентов функции потерь по всем весам сети с помощью **правила цепочки** (chain rule).

**Идея:**

1. Выполнить **forward pass** — получить предсказания и значение loss.
2. Вычислить градиент loss на **выходном слое**.
3. Распространить градиент **назад** по слоям, перемножая локальные производные.
4. Обновить веса градиентным спуском.

**Обновление весов (SGD):**

$$
\mathbf{W} \leftarrow \mathbf{W} - \eta \frac{\partial \mathcal{L}}{\partial \mathbf{W}}
$$

где $\eta$ — **learning rate** (скорость обучения).

**Проблемы при обучении:**

- **Переобучение (overfitting)** — модель запоминает обучающую выборку, плохо обобщает. Решение: регуляризация, dropout (изучим позже), early stopping, больше данных.
- **Несходимость** — слишком большой learning rate. Решение: уменьшить $\eta$, нормализовать признаки.
- **Медленная сходимость** — слишком маленький $\eta$ или плохая инициализация.

---

### 1.7. Подготовка данных для нейросети

Перед обучением необходимо:

1. **Разделить данные** на train / validation / test (например, 70% / 15% / 15%).
2. **Нормализовать признаки** — привести к нулевому среднему и единичной дисперсии (StandardScaler). Без нормализации градиентный спуск сходится медленно или не сходится.
3. **Закодировать метки классов** — one-hot encoding для cross-entropy.
4. **Обрабатывать пропуски и категориальные признаки** — в реальных задачах это обязательный этап (в этой работе вы выполните его самостоятельно).

**Важно:** нормализацию нужно **обучить только на train** и применить к val/test — иначе произойдёт **утечка данных** (data leakage).

---

### 1.8. Практический кейс: скрининг сердечно-сосудистого риска

**Скрининг сердечно-сосудистого риска** — задача оценки вероятности наличия ишемической болезни сердца (ИБС) по результатам обследования. Такие модели используются как **вспомогательный инструмент** для врачей: они помогают выделить группу пациентов с повышенным риском и направить их на дополнительное обследование.

**Признаки** (пример): возраст, пол, тип боли в груди, артериальное давление, уровень холестерина, максимальный пульс при нагрузке, изменения на ЭКГ.

**Целевая переменная:** бинарная метка — `0` (ИБС не выявлена) / `1` (ИБС выявлена).

**Метрики качества:**

- **Accuracy** — доля верных предсказаний (может быть обманчивой при дисбалансе классов).
- **Precision** — среди предсказанных «больных» сколько реально больны.
- **Recall (Sensitivity)** — среди реально больных пациентов сколько модель выявила. **Критически важен** в медицине: пропустить больного (ложно-отрицательный результат) опаснее, чем отправить здорового на дообследование.
- **F1-score** — гармоническое среднее precision и recall.
- **ROC-AUC** — площадь под ROC-кривой; показывает качество ранжирования независимо от порога.

В этой лабораторной работе вы построите MLP **без использования готовых DL-фреймворков** (PyTorch/TensorFlow), чтобы понять внутреннее устройство нейросети. Это фундамент для всех последующих работ.

---

## 2. Цель работы

1. Понять математическую модель многослойного перceptron и алгоритм backpropagation.
2. Реализовать MLP на NumPy с нуля.
3. Применить модель к **реальной задаче медицинской аналитики** — предсказание риска ИБС по результатам обследования.
4. Сравнить качество с **логистической регрессией** (baseline из первого семестра).
5. Научиться интерпретировать метрики классификации в прикладном контексте.

---

## 3. Задача

Медицинский центр предоставил обезличенный датасет результатов кардиологического обследования (**Heart Disease Dataset, Cleveland subset**, UCI Machine Learning Repository). Каждая строка — один пациент, целевая переменная:

- `0` — ишемическая болезнь сердца **не выявлена**;
- `1` — ишемическая болезнь сердца **выявлена**.

**Ваша задача:** построить MLP, предсказывающий наличие ИБС, добиться **ROC-AUC ≥ 0.72** на тестовой выборке и подготовить отчёт с анализом результатов.

> **Примечание:** в оригинальном датасете целевая переменная принимает значения 0–4 (степень выраженности болезни). В работе мы преобразуем её в бинарную: `0` → 0 (здоров), `1–4` → 1 (болен).

**Описание признаков:**

| Признак | Описание |
|---------|----------|
| `age` | Возраст (лет) |
| `sex` | Пол (1 = мужской, 0 = женский) |
| `cp` | Тип боли в груди (1–4) |
| `trestbps` | Артериальное давление в покое (мм рт. ст.) |
| `chol` | Холестерин сыворотки (мг/дл) |
| `fbs` | Сахар натощак > 120 мг/дл (1 = да, 0 = нет) |
| `restecg` | Результаты ЭКГ в покое (0–2) |
| `thalach` | Максимальный пульс при нагрузке |
| `exang` | Стенокардия при нагрузке (1 = да, 0 = нет) |
| `oldpeak` | Депрессия сегмента ST при нагрузке |
| `slope` | Наклон пика ST при нагрузке (1–3) |
| `ca` | Число major vessels (0–3) |
| `thal` | Результат таллиевого теста (3 = normal, 6 = fixed, 7 = reversible) |

---

## 4. Ход выполнения работы

### Шаг 1. Подготовка среды и загрузка данных

Создайте новый notebook в Google Colab. Установите зависимости (если нужно) и загрузите данные.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, RocCurveDisplay
)

np.random.seed(42)
```

**Загрузка Cleveland Heart Disease Dataset.** Датасет не имеет заголовков; пропуски обозначены символом `?`. Описание — в [документации UCI](https://archive.ics.uci.edu/ml/datasets/heart+disease).

```python
DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)

COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
]

raw = pd.read_csv(DATA_URL, header=None, names=COLUMNS, na_values="?")
print(f"Размер датасета (до очистки): {raw.shape}")
raw.head()
```

**Задание 1.1.** Изучите датасет: выведите `info()`, число пропусков (`isnull().sum()`), распределение целевой переменной. Опишите дисбаланс классов в 2–3 предложениях.

```python
# Удаление строк с пропусками (их немного — 6 из 303)
raw = raw.dropna()
print(f"Размер после удаления пропусков: {raw.shape}")

# Бинаризация целевой переменной: 0 → 0 (здоров), 1–4 → 1 (болен)
raw["target"] = (raw["target"] > 0).astype(int)

print(raw["target"].value_counts(normalize=True).round(3))
```

---

### Шаг 2. Предобработка признаков

Датасет содержит **числовые** и **категориальные** (закодированные числами) признаки. Нормализуйте числовые столбцы, затем выполните one-hot encoding категориальных.

```python
def preprocess_data(df):
    """
    Подготавливает признаки: нормализация числовых + one-hot категориальных.
    Возвращает X (np.array), y (np.array), numeric_cols.
    """
    df = df.copy()
    y = df["target"].values.astype(np.int64)

    numeric_cols = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    categorical_cols = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]

    # TODO (Задание 2.1): one-hot encoding для categorical_cols
    # Используйте pd.get_dummies(..., drop_first=True)
    X_num = df[numeric_cols].values.astype(np.float64)
    X_cat = ...  # ваш код

    X = np.hstack([X_num, X_cat.values]).astype(np.float64)
    return X, y, numeric_cols


X, y, numeric_cols = preprocess_data(raw)
print(f"Размерность признаков после encoding: {X.shape}")
```

**Разбиение на выборки и нормализация:**

```python
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

scaler = StandardScaler()
num_features = len(numeric_cols)

# TODO (Задание 2.2): fit scaler на числовых признаках train,
# transform train/val/test (только первые num_features столбцов)
X_train[:, :num_features] = ...
X_val[:, :num_features] = ...
X_test[:, :num_features] = ...
```

**One-hot encoding меток** (для cross-entropy):

```python
def to_one_hot(y, num_classes=2):
    """Преобразует вектор меток (m,) в one-hot матрицу (m, num_classes)."""
    m = y.shape[0]
    one_hot = np.zeros((m, num_classes))
    one_hot[np.arange(m), y] = 1
    return one_hot

y_train_oh = to_one_hot(y_train)
y_val_oh = to_one_hot(y_val)
# y_test оставьте в исходном виде для sklearn-метрик
```

---

### Шаг 3. Реализация функций активации

```python
def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))

def softmax(z):
    """
    z: (batch_size, num_classes)
    Возвращает вероятности классов, сумма по строке = 1.
    """
    # TODO (Задание 3.1): реализуйте numerically stable softmax
    # Подсказка: вычтите max(z) по оси классов перед exp
    z_shifted = ...
    exp_z = ...
    return ...
```

**Проверка:**

```python
test_z = np.array([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]])
probs = softmax(test_z)
assert np.allclose(probs.sum(axis=1), 1.0), "Softmax: сумма по строке должна быть 1"
print("Softmax OK:", probs)
```

---

### Шаг 4. Класс MLP

Реализуйте класс с двумя обучаемыми слоями: **скрытый (ReLU) + выходной (softmax)**.

```python
class MLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.lr = learning_rate
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, output_size))

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = relu(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = softmax(self.z2)
        return self.a2

    def compute_loss(self, y_pred, y_true):
        m = y_true.shape[0]
        # TODO (Задание 4.1): реализуйте cross-entropy с epsilon для log(0)
        epsilon = 1e-9
        loss = ...
        return loss

    def backward(self, X, y_true):
        m = X.shape[0]

        dz2 = self.a2 - y_true

        dW2 = (self.a1.T @ dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m

        da1 = dz2 @ self.W2.T
        dz1 = da1 * relu_derivative(self.z1)

        # TODO (Задание 4.2): вычислите dW1 и db1
        dW1 = ...
        db1 = ...

        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

    def predict(self, X):
        probs = self.forward(X)
        return np.argmax(probs, axis=1)

    def predict_proba(self, X):
        return self.forward(X)[:, 1]
```

---

### Шаг 5. Цикл обучения

```python
def train_mlp(model, X_train, y_train_oh, X_val, y_val,
              epochs=500, batch_size=32, verbose=True):
    history = {"train_loss": [], "val_auc": []}
    m = X_train.shape[0]

    for epoch in range(epochs):
        indices = np.random.permutation(m)
        X_shuffled = X_train[indices]
        y_shuffled = y_train_oh[indices]

        epoch_loss = 0.0
        num_batches = 0

        for start in range(0, m, batch_size):
            X_batch = X_shuffled[start:start + batch_size]
            y_batch = y_shuffled[start:start + batch_size]

            y_pred = model.forward(X_batch)
            loss = model.compute_loss(y_pred, y_batch)
            model.backward(X_batch, y_batch)

            epoch_loss += loss
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        val_probs = model.predict_proba(X_val)
        val_auc = roc_auc_score(y_val, val_probs)

        history["train_loss"].append(avg_loss)
        history["val_auc"].append(val_auc)

        if verbose and epoch % 50 == 0:
            print(f"Epoch {epoch:4d} | Loss: {avg_loss:.4f} | Val AUC: {val_auc:.4f}")

    return history
```

**Запуск обучения:**

```python
input_size = X_train.shape[1]
model = MLP(input_size=input_size, hidden_size=16, output_size=2, learning_rate=0.1)

history = train_mlp(model, X_train, y_train_oh, X_val, y_val, epochs=500, batch_size=32)
```

**Задание 5.1.** Постройте графики `train_loss` и `val_auc` от номера эпохи. Есть ли признаки переобучения?

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
# TODO: постройте графики
plt.tight_layout()
plt.show()
```

**Задание 5.2.** Подберите `hidden_size` и `learning_rate` (минимум 3 эксперимента). Заполните таблицу:

| hidden_size | learning_rate | Val AUC | Test AUC |
|-------------|---------------|---------|----------|
| 8           | 0.1           |         |          |
| 16          | 0.1           |         |          |
| 32          | 0.05          |         |          |

---

### Шаг 6. Оценка на тестовой выборке

```python
y_test_pred = model.predict(X_test)
y_test_proba = model.predict_proba(X_test)

print("=== MLP — Test Metrics ===")
print(f"Accuracy:  {accuracy_score(y_test, y_test_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_test_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_test_pred):.4f}")
print(f"F1:        {f1_score(y_test, y_test_pred):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_test_proba):.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_test_pred))
```

---

### Шаг 7. Baseline — логистическая регрессия

Сравните MLP с моделью из первого семестра:

```python
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)

y_lr_proba = lr_model.predict_proba(X_test)[:, 1]
print("=== Logistic Regression — Test Metrics ===")
print(f"ROC-AUC: {roc_auc_score(y_test, y_lr_proba):.4f}")
print(f"F1:      {f1_score(y_test, lr_model.predict(X_test)):.4f}")
```

**Задание 7.1.** Постройте ROC-кривые обеих моделей на одном графике (`RocCurveDisplay.from_predictions`).

**Задание 7.2.** Напишите 5–7 предложений: какая модель лучше и почему? Учитывайте, что в медицинской задаче **Recall** (чувствительность) для класса «болен» важнее Accuracy — пропустить больного пациента опаснее, чем ошибочно отправить здорового на дообследование.

---

### Шаг 8. Анализ ошибок (бонус, +10% к оценке)

```python
# Найдите 5 пациентов, где MLP ошибся, а Logistic Regression — нет (и наоборот)
# TODO: ваш код

# Выведите их признаки и предсказанные вероятности
# Подсказка: используйте raw.iloc[...] для интерпретации признаков
```

---

## 5. Содержание отчёта

Отчёт оформляется в PDF или Word и содержит:

1. **Титульный лист** — дисциплина, номер работы, ФИО, группа, дата.
2. **Цель работы** — кратко своими словами.
3. **Описание датасета** — источник, число объектов и признаков, распределение классов, комментарий о дисбалансе, обработка пропусков.
4. **Описание реализации** — архитектура MLP (число слоёв, нейронов, функции активации, loss, optimizer), блок-схема forward/backward (можно нарисовать от руки и вставить фото).
5. **Эксперименты** — таблица подбора гиперпараметров (шаг 5.2).
6. **Результаты** — метрики MLP и Logistic Regression на test; ROC-кривые; confusion matrix.
7. **Графики обучения** — loss и val AUC по эпохам.
8. **Выводы** — 3–5 пунктов: что удалось, где MLP лучше/хуже baseline, что бы улучшили.
9. **Ответы на контрольные вопросы** — см. раздел 6.
10. **Приложение** — ссылка на Colab-notebook или листинг ключевых функций.

**Объём:** 6–10 страниц (без приложения).

---

## 6. Контрольные вопросы

1. Чем один нейрон с sigmoid-активацией и cross-entropy loss отличается от логистической регрессии? А чем совпадает?
2. Зачем нужна функция активации в скрытых слоях? Что будет, если заменить ReLU на линейную функцию $f(z) = z$?
3. Объясните алгоритм backpropagation своими словами. Почему градиент на выходном слое при softmax + cross-entropy равен $(\hat{y} - y)$?
4. Что такое learning rate и как он влияет на обучение? Приведите примеры «слишком большого» и «слишком маленького» learning rate по вашим экспериментам.
5. Почему признаки нужно нормализовать перед обучением нейросети?
6. В задаче скрининга ИБС врач хочет минимизировать пропуск больных пациентов. Какую метрику (Accuracy, Precision, Recall, F1) вы бы оптимизировали в первую очередь? Почему?
7. Что такое переобучение? Какие признаки переобучения вы наблюдали (или не наблюдали) на графиках обучения?
8. Объясните, почему мы используем mini-batch SGD, а не обновляем веса один раз за эпоху (batch GD) или после каждого объекта (online GD).

---

## Полезные ссылки

- [Heart Disease Dataset (UCI)](https://archive.ics.uci.edu/ml/datasets/heart+disease)
- [Colab: Getting Started](https://colab.research.google.com/)
- Глава 6: Deep Learning (Goodfellow et al.) — feedforward networks
- [d2l.ai — Multilayer Perceptrons](https://d2l.ai/chapter_multilayer-perceptrons/)
