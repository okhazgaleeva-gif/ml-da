# Лабораторная работа 2

## Обучение MLP в PyTorch — классификация изображений

**Дисциплина:** Машинное обучение и анализ данных  
**Семестр:** 2  
**Инструменты:** Python 3, PyTorch, torchvision, NumPy, Matplotlib, scikit-learn  
**Среда:** Google Colab (рекомендуется, GPU не обязателен, но ускоряет обучение)

---

## 1. Теоретические сведения

### 1.1. От NumPy к PyTorch

В лабораторной работе 1 вы реализовали MLP «с нуля» на NumPy: вручную написали forward pass, backward pass и цикл обучения. Это дало понимание внутренней механики, но такой подход плохо масштабируется: для реальных сетей с десятками слоёв и миллионов параметров ручное вычисление градиентов становится трудоёмким и подверженным ошибкам.

PyTorch решает эту проблему за счёт **автоматического дифференцирования (autograd)**. Вместо ручного вывода формул для градиентов вы описываете прямой проход сети, а PyTorch строит вычислительный граф и автоматически вычисляет все градиенты при вызове `.backward()`. Это позволяет сосредоточиться на архитектуре и экспериментах, а не на выводе производных.

Ключевые абстракции PyTorch:
- **Tensor** — многомерный массив, аналог `np.ndarray`, но с поддержкой автодифференцирования и возможности работы на GPU.
- **nn.Module** — базовый класс для всех нейронных сетей. Содержит параметры (`nn.Parameter`) и методы `forward()`.
- **nn.functional** — функциональный интерфейс для операций (активации, loss-функции, dropout и т.д.).
- **torch.optim** — модуль с реализациями оптимизаторов (SGD, Adam, AdamW и др.).
- **torch.utils.data** — `Dataset` и `DataLoader` для эффективной загрузки и батчинга данных.

### 1.2. Оптимизаторы

В лабораторной 1 вы использовали простейший стохастический градиентный спуск (SGD):

$$W \leftarrow W - \eta \frac{\partial L}{\partial W}$$

На практике чистый SGD часто сходится медленно и чувствителен к выбору learning rate. Современные оптимизаторы добавляют в это правило дополнительные механизмы:

**SGD с моментом (Momentum).** Идея: накапливать «инерцию» движения по параметрам, чтобы сглаживать колебания и ускоряться в согласованных направлениях.

$$v_t = \beta v_{t-1} + (1 - \beta) \nabla L, \quad W \leftarrow W - \eta v_t$$

где $\beta$ — коэффициент момента (обычно 0.9). Momentum помогает «прокатываться» через небольшие локальные минимумы и ускоряет сходимость в оврагах.

**Adam (Adaptive Moment Estimation).** Комбинирует идею момента с адаптивным learning rate для каждого параметра. Хранит оценки первого и второго моментов градиента:

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t, \quad v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}, \quad W \leftarrow W - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

Adam обычно требует меньше настройки learning rate и хорошо работает «из коробки». AdamW — это вариант Adam с корректно разделённым weight decay (L2-регуляризация применяется отдельно от адаптивного шага), что часто даёт лучшую обобщающую способность.

**Практическое правило:** для большинства задач Adam/AdamW — хорошая отправная точка; SGD с моментом может дать лучший финальный результат при тщательном подборе learning rate и schedule, но требует больше экспериментов.

### 1.3. Регуляризация

**Dropout.** Во время обучения каждый нейрон с вероятностью $p$ «выключается» (его выход обнуляется), а оставшиеся масштабируются на $1/(1-p)$. Это предотвращает co-adaptation нейронов и действует как ансамбль подмоделей. На этапе инференса dropout отключается. Типичные значения $p$: 0.2–0.5 для полносвязных слоёв.

**Batch Normalization (BatchNorm).**

*Этот подраздел — дополнительный материал. В лекции 2 BatchNorm не разбирается, но в лабораторной работе он используется. Прочитайте внимательно — без понимания этого механизма шаги 5 и 6 выполнить осмысленно нельзя.*

**Проблема, которую решает BatchNorm.** При обучении глубокой сети распределение входов каждого слоя постоянно меняется: предыдущий слой обновляет свои веса, и следующий слой вынужден подстраиваться под новый масштаб и смещение активаций. Это явление называют *internal covariate shift*. Оно замедляет обучение и делает сеть чувствительной к выбору learning rate. Особенно сильно это проявляется в полносвязных сетях с насыщающимися активациями (sigmoid, tanh): если вход слоя смещён в область насыщения, градиент через него почти не проходит (см. лекцию 2, слайды 7 и 15).

**Идея.** Нормализовать активации внутри каждого мини-батча перед подачей в следующий слой: вычесть среднее по батчу и поделить на стандартное отклонение. Затем применить обучаемые масштаб $\gamma$ и сдвиг $\beta$, чтобы сеть могла восстановить любое нужное распределение, если нормализация окажется вредной.

**Формула.** Для батча $B = \{x_1, \dots, x_m\}$ и каждого канала (признака):

$$\mu_B = \frac{1}{m}\sum_{i=1}^{m} x_i, \qquad \sigma_B^2 = \frac{1}{m}\sum_{i=1}^{m}(x_i - \mu_B)^2$$

$$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}, \qquad y_i = \gamma \hat{x}_i + \beta$$

где $\epsilon$ — маленькая константа для численной стабильности (обычно $10^{-5}$), а $\gamma$ и $\beta$ — обучаемые параметры (по одному на признак).

**Train vs eval — ключевое различие.** На обучении статистики $\mu_B$ и $\sigma_B^2$ считаются **по текущему батчу**. На инференсе батча может не быть вовсе (одно изображение), и зависеть от случайного состава батча нельзя. Поэтому PyTorch во время обучения параллельно накапливает *running estimates* — экспоненциально сглаженные средние по всем виденным батчам — и на eval использует именно их. Технически это выглядит так:

 ```python
 # внутри nn.BatchNorm1d
 if self.training:
     mean = x.mean(dim=0)
     var  = x.var(dim=0, unbiased=False)
     self.running_mean = (1 - momentum) * self.running_mean + momentum * mean.detach()
     self.running_var  = (1 - momentum) * self.running_var  + momentum * var.detach()
 else:
     mean = self.running_mean
     var  = self.running_var
 ```

Отсюда — обязательное правило: **перед валидацией и тестом всегда вызывайте `model.eval()`**. Если забыть, BatchNorm продолжит использовать статистики батча, и метрики будут нестабильными и завышенными (особенно на маленьких батчах).

**Почему это работает как регуляризация.** Нормализация по батчу добавляет шум: статистики каждого батча немного отличаются, поэтому активации одного и того же примера в разных батчах слегка разные. Это похоже на эффект dropout — сеть не может «зазубрить» конкретные значения активаций. Дополнительно BatchNorm сглаживает ландшафт функции потерь (уменьшает кривизну), что позволяет использовать больший learning rate без расходимости.

**Куда ставить BatchNorm.** Классический порядок в полносвязной сети: `Linear → BatchNorm → ReLU → Dropout`. То есть BatchNorm — **до** активации. Некоторые современные архитектуры ставят после (`Linear → ReLU → BatchNorm`), но для MLP на Fashion-MNIST первый вариант работает стабильнее. В коде лабораторной это уже реализовано:

 ```python
 layers.append(nn.Linear(prev_size, h))
 if use_batchnorm:
     layers.append(nn.BatchNorm1d(h))   # до ReLU
 layers.append(nn.ReLU())
 if dropout_rate > 0:
     layers.append(nn.Dropout(dropout_rate))
 ```

**Известное ограничение.** BatchNorm плохо работает при очень маленьком размере батча (меньше 8–16): статистики становятся шумными. В лабораторной используется `batch_size=64` для train, поэтому проблемы не будет. Но при валидации, если батч равен 1, `nn.BatchNorm1d` в режиме `train()` выбросит ошибку `Expected more than 1 value per channel` — ещё одна причина всегда вызывать `model.eval()` перед валидацией.


**L2-регуляризация (weight decay).** Добавляет к функции потерь штраф за большие веса: $L_{\text{total}} = L + \lambda \|W\|^2$. В PyTorch задаётся через параметр `weight_decay` в оптимизаторе.

**Early stopping.** Прекращение обучения, когда метрика на валидации перестаёт улучшаться. Простейшая реализация — сохранение лучшей модели по val loss/accuracy и остановка при отсутствии улучшений в течение $N$ эпох.

### 1.4. Dataset и DataLoader

PyTorch предоставляет два ключевых класса для работы с данными:

- **Dataset** — абстракция, которая знает, как получить один объект по индексу (`__getitem__`) и сколько объектов всего (`__len__`). Для стандартных датасетов (MNIST, Fashion-MNIST, CIFAR-10) готовые реализации есть в `torchvision.datasets`.
- **DataLoader** — обёртка над Dataset, которая автоматически разбивает данные на батчи, перемешивает их (если `shuffle=True`), загружает в параллельных потоках (`num_workers`) и позволяет итерироваться по батчам в цикле обучения.

Типичный цикл обучения в PyTorch:

```python
for epoch in range(num_epochs):
    model.train()
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
    
    model.eval()
    with torch.no_grad():
        # валидация
        ...
```

Обратите внимание на три обязательных шага в каждом батче: `optimizer.zero_grad()` (обнуление старых градиентов), `loss.backward()` (вычисление градиентов) и `optimizer.step()` (обновление параметров). Пропуск `zero_grad()` приведёт к накоплению градиентов с предыдущих батчей.

### 1.5. Классификация изображений с помощью MLP

Fashion-MNIST — датасет из 70 000 изображений одежды (60 000 train, 10 000 test) размером 28×28 пикселей в оттенках серого. 10 классов: футболка, брюки, пуловер, платье, пальто, сандалии, рубашка, кроссовки, сумка, ботинки. Это более сложная замена MNIST: классы визуально ближе друг к другу, поэтому accuracy ниже.

Для MLP каждое изображение «разворачивается» в вектор длины 784 (28×28). Архитектура: `784 → hidden → ... → 10`. Выходной слой — 10 логитов (без softmax: `nn.CrossEntropyLoss` в PyTorch ожидает «сырые» логиты и сам применяет log-softmax внутри).

CIFAR-10 — более сложный датасет: 60 000 цветных изображений 32×32×3, 10 классов (самолёт, автомобиль, птица, кот, олень, собака, лягушка, лошадь, корабль, грузовик). Для MLP изображение разворачивается в вектор длины 3072. MLP на CIFAR-10 показывает заметно более низкое качество, чем CNN, но хорошо демонстрирует ограничения полносвязных архитектур для изображений.

**Рекомендация:** в этой работе используйте Fashion-MNIST как основной датасет (быстрее обучается, стабильные результаты). CIFAR-10 можно взять как дополнительный эксперимент (бонус).

---

## 2. Практический кейс

Задача: классифицировать изображения одежды по 10 категориям. Такая модель может использоваться в приложениях электронной коммерции для автоматической категоризации товаров, в системах сортировки или как компонент рекомендательных систем.

Мы обучим MLP в PyTorch, сравним несколько оптимизаторов, исследуем влияние dropout и BatchNorm, научимся логировать метрики и визуализировать процесс обучения.

---

## 3. Ход выполнения работы

### Шаг 1. Подготовка среды и загрузка данных

Создайте новый notebook в Google Colab. Установите зависимости (в Colab PyTorch уже установлен, но проверьте версию):

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, ConfusionMatrixDisplay)
import time
from copy import deepcopy

# Фиксация seed для воспроизводимости
torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"PyTorch version: {torch.__version__}")
```

Загрузка Fashion-MNIST:

```python
# Трансформации: ToTensor переводит PIL-изображение в тензор [0,1] и меняет порядок осей
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,))  # mean/std Fashion-MNIST
])

train_dataset = torchvision.datasets.FashionMNIST(
    root="./data", train=True, download=True, transform=transform
)
test_dataset = torchvision.datasets.FashionMNIST(
    root="./data", train=False, download=True, transform=transform
)

CLASSES = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

print(f"Train size: {len(train_dataset)}")
print(f"Test size:  {len(test_dataset)}")
print(f"Image shape: {train_dataset[0][0].shape}")
```

**Задание 1.1.** Выведите 9 случайных изображений из train_dataset в виде сетки 3×3 с указанием истинных классов. Визуально определите, какие классы легко перепутать.

```python
# TODO: ваш код для визуализации
fig, axes = plt.subplots(3, 3, figsize=(9, 9))
for ax in axes.flat:
    idx = np.random.randint(len(train_dataset))
    img, label = train_dataset[idx]
    ax.imshow(img.squeeze(), cmap="gray")
    ax.set_title(CLASSES[label])
    ax.axis("off")
plt.tight_layout()
plt.show()
```

**Задание 1.2.** Разделите train_dataset на train и validation (например, 90% / 10%). Используйте `Subset` и `torch.Generator` для воспроизводимости. Создайте `DataLoader` для train, val и test с `batch_size=64`, `shuffle=True` для train и `shuffle=False` для val/test.

```python
# TODO: ваш код для split и DataLoader
from torch.utils.data import random_split

train_size = int(0.9 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_subset, val_subset = random_split(
    train_dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(train_subset, batch_size=64, shuffle=True, num_workers=2)
val_loader   = DataLoader(val_subset, batch_size=256, shuffle=False, num_workers=2)
test_loader  = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=2)
```

### Шаг 2. Определение архитектуры MLP

Реализуйте класс `MLP` с двумя скрытыми слоями. Используйте `nn.Module`, `nn.Linear`, `nn.ReLU`, `nn.Dropout`, `nn.BatchNorm1d`.

```python
class MLP(nn.Module):
    """
    Полносвязная сеть для классификации изображений.
    Архитектура: Flatten → Linear → [BatchNorm] → ReLU → [Dropout] → ... → Linear
    """
    def __init__(self, input_size=784, hidden_sizes=(256, 128), num_classes=10,
                 dropout_rate=0.0, use_batchnorm=False):
        super().__init__()
        layers = []
        prev_size = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(prev_size, h))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            prev_size = h
        layers.append(nn.Linear(prev_size, num_classes))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten: (B, 1, 28, 28) → (B, 784)
        return self.net(x)
```

**Задание 2.1.** Создайте модель с `hidden_sizes=(256, 128)`, `dropout_rate=0.0`, `use_batchnorm=False`. Выведите число обучаемых параметров:

```python
model = MLP().to(device)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable parameters: {total_params:,}")
```

**Задание 2.2.** Ответьте письменно: сколько параметров в первом слое? Во втором? В выходном? Сравните с MLP из лабораторной 1 (там было ~13 000 параметров). Почему для изображений нужно больше параметров?

### Шаг 3. Функция потерь и цикл обучения

`nn.CrossEntropyLoss` объединяет `LogSoftmax` и `NLLLoss`. Она ожидает на входе логиты (без softmax) и целочисленные метки (не one-hot).

```python
criterion = nn.CrossEntropyLoss()
```

Реализуйте функцию обучения одной эпохи и функцию оценки:

```python
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * X_batch.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)
    return running_loss / total, correct / total

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        running_loss += loss.item() * X_batch.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())
    return running_loss / total, correct / total, np.array(all_preds), np.array(all_labels)
```

Функция полного обучения с early stopping и сохранением лучшей модели:

```python
def fit(model, train_loader, val_loader, criterion, optimizer,
        epochs=30, patience=5, device=device, verbose=True):
    history = {"train_loss": [], "val_loss": [],
               "train_acc": [], "val_acc": []}
    best_val_loss = float("inf")
    best_weights = deepcopy(model.state_dict())
    epochs_no_improve = 0
    
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        dt = time.time() - t0
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        if verbose:
            print(f"Epoch {epoch:3d} | {dt:.1f}s | "
                  f"Train loss {train_loss:.4f} acc {train_acc:.4f} | "
                  f"Val loss {val_loss:.4f} acc {val_acc:.4f}")
        
        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch} (best val loss {best_val_loss:.4f})")
            break
    
    model.load_state_dict(best_weights)
    return history
```

**Задание 3.1.** Обучите базовую модель (без dropout и BatchNorm) с оптимизатором SGD (lr=0.01, momentum=0.0) в течение 30 эпох. Постройте графики train/val loss и accuracy.

```python
base_model = MLP(dropout_rate=0.0, use_batchnorm=False).to(device)
optimizer_sgd = optim.SGD(base_model.parameters(), lr=0.01)
history_sgd = fit(base_model, train_loader, val_loader, criterion, optimizer_sgd,
                  epochs=30, patience=7)
```

```python
def plot_history(history, title=""):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title(f"{title} — Loss"); axes[0].legend()
    axes[1].plot(history["train_acc"], label="train")
    axes[1].plot(history["val_acc"], label="val")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
    axes[1].set_title(f"{title} — Accuracy"); axes[1].legend()
    plt.tight_layout(); plt.show()

plot_history(history_sgd, "SGD")
```

**Задание 3.2.** Ответьте письменно: наблюдаете ли вы переобучение? В какой момент train accuracy начинает значительно опережать val accuracy? Сделайте вывод о необходимости регуляризации.

### Шаг 3.5. Сравнение функций потерь: MSE vs Cross-Entropy

В лекции 2 (слайды 7–9) разобрано, почему для задачи классификации MSE плохо сочетается с сигмоидой/softmax на выходе, а Cross-Entropy — хорошо. Проверим это экспериментально.

**Напоминание теории.**

 - **MSE + sigmoid/softmax.** Градиент по логиту $z$ выглядит как
   $$\frac{\partial L}{\partial z} = (\hat{y} - y) \cdot \sigma'(z),$$
   где $\sigma'(z) \to 0$ при насыщении. Если сеть уверенно ошибается ($\hat{y} = 0.01$, $y = 1$), произведение даёт почти ноль — сеть не учится. Это эффект **затухающего градиента**.
 - **Cross-Entropy + softmax.** Градиент упрощается до
   $$\frac{\partial L}{\partial z} = \hat{y} - y,$$
   без множителя $\sigma'(z)$. При сильной ошибке градиент большой, сеть быстро исправляется.

 **Задание 3.3.** Создайте две одинаковые модели `MLP(dropout_rate=0.0, use_batchnorm=False)` с одинаковым seed. Обучите их с одним и тем же оптимизатором `SGD(lr=0.01)` по 15 эпох:

 1. **Модель A** — с `nn.CrossEntropyLoss()` (как в задании 3.1).
 2. **Модель B** — с `nn.MSELoss()`. Так как MSE ожидает на входе вероятности, а не логиты, добавьте в модель `nn.Softmax(dim=1)` на выходе. One-hot метки получите через `torch.nn.functional.one_hot(y_batch, num_classes=10).float()`.

 Скелет для модели B:

 ```python
 class MLPWithSoftmax(MLP):
     def forward(self, x):
         logits = super().forward(x)
         return torch.softmax(logits, dim=1)

 def train_epoch_mse(model, loader, optimizer, device):
     model.train()
     running_loss = 0.0
     correct = 0
     total = 0
     for X_batch, y_batch in loader:
         X_batch, y_batch = X_batch.to(device), y_batch.to(device)
         optimizer.zero_grad()
         probs = model(X_batch)
         y_onehot = torch.nn.functional.one_hot(y_batch, num_classes=10).float()
         loss = nn.MSELoss()(probs, y_onehot)
         loss.backward()
         optimizer.step()
         running_loss += loss.item() * X_batch.size(0)
         preds = probs.argmax(dim=1)
         correct += (preds == y_batch).sum().item()
         total += y_batch.size(0)
     return running_loss / total, correct / total
 ```

 **Задание 3.4.** Постройте на одном графике:
 - `train_loss` для обеих моделей (по эпохам),
 - `val_accuracy` для обеих моделей (по эпохам).

 Заполните таблицу:

 | Функция потерь | Train Loss (эпоха 15) | Val Acc (эпоха 15) | Best Val Acc |
 |---|---|---|---|
 | Cross-Entropy | | | |
 | MSE + Softmax | | | |

 **Задание 3.5 (письменно).** Ответьте на вопросы, опираясь на формулы градиентов из лекции 2 (слайды 7–9):

 1. Какая модель сходится быстрее на первых 3–5 эпохах? Совпадает ли это с теоретическим предсказанием?
 2. Какая модель достигает более высокой accuracy к 15-й эпохе?
 3. Посмотрите на график `train_loss` для модели B: убывает ли MSE так же быстро, как CE? Обратите внимание, что значения MSE и CE **несравнимы по величине** — MSE всегда в диапазоне $[0, 2]$ для one-hot, CE неограничена сверху. Поэтому важен не абсолютный уровень, а **форма кривой** и итоговая accuracy.
 4. Объясните в 3–4 предложениях, почему MSE + softmax может «застревать» на ранней стадии обучения, когда модель уверенно ошибается. Используйте формулу $\partial L / \partial z = (\hat{y} - y) \cdot \sigma'(z)$.
 5. Почему в PyTorch `nn.CrossEntropyLoss` принимает на вход **логиты**, а не вероятности? Что произойдёт, если добавить `nn.Softmax` перед `CrossEntropyLoss`?



### Шаг 4. Сравнение оптимизаторов

Исследуйте четыре оптимизатора: SGD (без момента), SGD с моментом 0.9, Adam и AdamW. Для каждого используйте одинаковую архитектуру (без dropout/BatchNorm) и одинаковое число эпох.

```python
def run_experiment(optimizer_class, optimizer_kwargs, model_kwargs,
                   epochs=30, patience=7, name=""):
    torch.manual_seed(42)
    model = MLP(**model_kwargs).to(device)
    optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)
    t0 = time.time()
    history = fit(model, train_loader, val_loader, criterion, optimizer,
                  epochs=epochs, patience=patience, verbose=False)
    total_time = time.time() - t0
    _, test_acc, _, _ = evaluate(model, test_loader, criterion, device)
    best_val_acc = max(history["val_acc"])
    print(f"{name:20s} | best val acc {best_val_acc:.4f} | "
          f"test acc {test_acc:.4f} | time {total_time:.1f}s")
    return history, test_acc
```

```python
experiments = [
    ("SGD (no momentum)", optim.SGD,    {"lr": 0.01},                        {}),
    ("SGD + momentum",    optim.SGD,    {"lr": 0.01, "momentum": 0.9},       {}),
    ("Adam",              optim.Adam,   {"lr": 1e-3},                        {}),
    ("AdamW",             optim.AdamW,  {"lr": 1e-3, "weight_decay": 1e-4},  {}),
]

results = {}
for name, opt_cls, opt_kwargs, model_kwargs in experiments:
    hist, test_acc = run_experiment(opt_cls, opt_kwargs, model_kwargs,
                                    epochs=30, name=name)
    results[name] = (hist, test_acc)
```

**Задание 4.1.** Постройте на одном графике val accuracy для всех четырёх оптимизаторов. Заполните таблицу:

| Оптимизатор | Best Val Acc | Test Acc | Время обучения (с) |
|---|---|---|---|
| SGD (no momentum) | | | |
| SGD + momentum | | | |
| Adam | | | |
| AdamW | | | |

**Задание 4.2.** Напишите 5–7 предложений: какой оптимизатор показал лучший результат? Совпадает ли это с вашими ожиданиями? Объясните, почему Adam/AdamW обычно сходятся быстрее на первых эпохах, а SGD с моментом может догнать их при большем числе эпох.

### Шаг 5. Влияние dropout и BatchNorm

Теперь зафиксируйте лучший оптимизатор (по результатам шага 4) и исследуйте влияние регуляризации. Проведите три эксперимента:

1. Без dropout и BatchNorm (baseline).
2. Только dropout (0.3) после каждого скрытого слоя.
3. Dropout (0.3) + BatchNorm после каждого скрытого слоя (до активации).

```python
best_opt_name = "AdamW"  # замените на лучший по вашим результатам
opt_map = {
    "SGD (no momentum)": (optim.SGD,    {"lr": 0.01}),
    "SGD + momentum":    (optim.SGD,    {"lr": 0.01, "momentum": 0.9}),
    "Adam":              (optim.Adam,   {"lr": 1e-3}),
    "AdamW":             (optim.AdamW,  {"lr": 1e-3, "weight_decay": 1e-4}),
}
opt_cls, opt_kwargs = opt_map[best_opt_name]

reg_experiments = [
    ("Baseline",        {"dropout_rate": 0.0, "use_batchnorm": False}),
    ("Dropout 0.3",     {"dropout_rate": 0.3, "use_batchnorm": False}),
    ("Dropout + BN",    {"dropout_rate": 0.3, "use_batchnorm": True}),
]

for name, model_kwargs in reg_experiments:
    hist, test_acc = run_experiment(opt_cls, opt_kwargs, model_kwargs,
                                    epochs=30, name=name)
    results[name] = (hist, test_acc)
```

**Задание 5.1.** Постройте графики train/val loss для трёх конфигураций. Есть ли признаки переобучения в baseline? Уменьшают ли dropout и BatchNorm разрыв между train и val?

**Задание 5.2.** Сравните test accuracy трёх конфигураций. Напишите 3–5 предложений: что дало наибольший прирост — dropout или BatchNorm? Почему BatchNorm особенно полезен для изображений?

**Задание 5.3 (бонус, +10%).** Попробуйте разные значения dropout_rate (0.1, 0.2, 0.3, 0.5). Постройте график test accuracy от dropout_rate.

### Шаг 6. Оценка лучшей модели на тестовой выборке

Выберите лучшую конфигурацию из шагов 4–5. Обучите её заново с фиксированным seed и оцените на тестовой выборке.

```python
# Замените на лучшую конфигурацию
best_model = MLP(dropout_rate=0.3, use_batchnorm=True).to(device)
best_optimizer = optim.AdamW(best_model.parameters(), lr=1e-3, weight_decay=1e-4)

history_best = fit(best_model, train_loader, val_loader, criterion,
                   best_optimizer, epochs=40, patience=7)

test_loss, test_acc, y_pred, y_true = evaluate(
    best_model, test_loader, criterion, device
)

print(f"Test Accuracy:  {test_acc:.4f}")
print(f"Test Loss:      {test_loss:.4f}")
print(f"Precision (macro): {precision_score(y_true, y_pred, average='macro'):.4f}")
print(f"Recall (macro):    {recall_score(y_true, y_pred, average='macro'):.4f}")
print(f"F1 (macro):        {f1_score(y_true, y_pred, average='macro'):.4f}")
```

**Задание 6.1.** Постройте confusion matrix. Какие классы модель путает чаще всего? Согласуется ли это с вашими наблюдениями из задания 1.1?

```python
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASSES)
fig, ax = plt.subplots(figsize=(10, 8))
disp.plot(ax=ax, cmap="Blues", xticks_rotation=45)
plt.title("Confusion Matrix — Best MLP")
plt.tight_layout()
plt.show()
```

**Задание 6.2.** Для двух самых проблемных пар классов (например, Shirt↔T-shirt, Pullover↔Coat) выведите по 5 примеров ошибок с изображениями и предсказанными вероятностями. Объясните, почему эти классы объективно сложны для MLP.

```python
# TODO: ваш код для визуализации ошибок
# Подсказка: найдите индексы, где y_true == class_a и y_pred == class_b,
# выведите соответствующие изображения из test_dataset
```

### Шаг 7. Логирование метрик (бонус, +5%)

Добавьте логирование в цикл обучения: сохраняйте train/val loss и accuracy каждую эпоху в CSV-файл или используйте `torch.utils.tensorboard` (если доступен). Постройте финальные графики обучения для лучшей модели.

```python
# Пример логирования в CSV
import csv

def fit_with_logging(model, train_loader, val_loader, criterion, optimizer,
                     epochs=30, patience=7, device=device, log_file="training_log.csv"):
    # TODO: модифицируйте fit(), добавив запись в CSV
    # Колонки: epoch, train_loss, train_acc, val_loss, val_acc, time_sec
    pass
```

---

## 4. Содержание отчёта

Отчёт оформляется в PDF или Word и содержит:

- **Титульный лист** — дисциплина, номер работы, ФИО, группа, дата.
- **Цель работы** — кратко своими словами.
- **Описание датасета** — Fashion-MNIST (или CIFAR-10, если использовали): источник, число объектов и классов, размер изображений, примеры визуализации.
- **Описание архитектуры** — число слоёв, нейронов, наличие dropout/BatchNorm, число параметров. Блок-схема forward pass.
- **Эксперименты** — таблица сравнения оптимизаторов (шаг 4.1), таблица влияния dropout/BatchNorm (шаг 5).
- **Результаты лучшей модели** — test accuracy, precision, recall, F1 (macro), confusion matrix, примеры ошибок.
- **Графики обучения** — loss и accuracy для лучшей конфигурации; сравнение val accuracy для разных оптимизаторов.
- **Выводы** — 3–5 пунктов: какой оптимизатор и какая регуляризация дали лучший результат, что было неожиданным, что бы вы улучшили.
- **Ответы на контрольные вопросы** — см. раздел 5.
- **Приложение** — ссылка на Colab-notebook или листинг ключевых функций.

Объём: 6–10 страниц (без приложения).

---

## 5. Контрольные вопросы

1. В чём принципиальное различие между реализацией MLP на NumPy (лабораторная 1) и на PyTorch? Какую роль играет autograd?

2. Объясните своими словами, что делает `optimizer.zero_grad()`. Что произойдёт, если его не вызывать?

3. Почему `nn.CrossEntropyLoss` ожидает на входе логиты, а не вероятности после softmax? Что произойдёт, если добавить softmax перед этой функцией потерь?

4. Чем Adam отличается от SGD с моментом? В каких случаях Adam может сходиться быстрее, а в каких SGD с моментом может дать лучший финальный результат?

5. Что такое BatchNorm? Почему поведение BatchNorm различается на train и eval? Что будет, если забыть вызвать `model.eval()` перед валидацией?

6. Как dropout влияет на обучение? Почему на этапе инференса dropout отключается? Что означало бы оставить dropout включённым на тесте?

7. Вы обучили MLP на Fashion-MNIST с test accuracy ~90%. Коллега предлагает увеличить число слоёв до 5 и число нейронов до 1024. Какие риски это несёт? Как бы вы проверили, улучшит ли это модель?

8. Почему MLP на CIFAR-10 показывает существенно более низкое качество, чем на Fashion-MNIST? Какая архитектура лучше подходит для CIFAR-10 и почему?

9. В чём разница между `model.train()` и `model.eval()`? Какие слои меняют своё поведение в зависимости от этого режима?

10. Предложите стратегию подбора learning rate для Adam. Почему «просто поставить 0.1» — плохая идея для Adam, хотя это разумное значение для SGD?

---

## 6. Полезные ссылки

- [Fashion-MNIST: A Novel Image Dataset for Benchmarking Machine Learning Algorithms](https://arxiv.org/abs/1708.07747)
- [PyTorch: Training a Classifier](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
- [PyTorch: Dataset and DataLoader](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)
- [PyTorch: Optimization](https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)
- [AdamW paper: Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)
- [Batch Normalization paper](https://arxiv.org/abs/1502.03167)
- [Dropout paper](https://jmlr.org/papers/v15/srivastava14a.html)
- [d2l.ai — Multilayer Perceptrons](https://d2l.ai/chapter_multilayer-perceptrons/)
- [d2l.ai — Linear Neural Networks for Classification](https://d2l.ai/chapter_linear-classification/)