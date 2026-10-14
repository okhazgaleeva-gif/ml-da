# Лабораторная работа №3

## Свёрточные нейронные сети (CNN) для классификации изображений

**Курс:** Машинное обучение и анализ данных
**Семестр:** 2
**Тема:** Свёрточные нейронные сети
**Среда выполнения:** Google Colab (рекомендуется) / Jupyter Notebook
**Язык:** Python 3

---

## Цель работы

Изучить архитектуру свёрточных нейронных сетей (CNN), освоить практические навыки обучения CNN с нуля и с применением transfer learning, научиться визуализировать работу сети (фильтры, feature maps, Grad-CAM) и оценивать качество модели на реальной задаче бинарной классификации изображений.

---

## Задачи

1. Разобраться, почему полносвязные сети (MLP) плохо подходят для изображений.
2. Реализовать свёрточную сеть с нуля на PyTorch.
3. Обучить CNN на датасете CIFAR-10 (задача «кошка vs собака»).
4. Применить transfer learning с предобученной ResNet-18.
5. Визуализировать фильтры первого слоя и карты признаков.
6. Построить тепловые карты Grad-CAM и проверить, куда «смотрит» модель.
7. Оценить качество модели с помощью метрик (accuracy, precision, recall, F1, AUC).

---

## Теоретическая справка

### Почему MLP не подходит для изображений

**Проблема 1. Взрывное число параметров.** Изображение 64×64×3 содержит 12 288 входов. Первый слой MLP с 256 нейронами требует 3 145 728 весов. Для изображения 256×256×3 потребовалось бы уже 50 миллионов весов только в одном слое.

**Проблема 2. Потеря пространственной структуры.** MLP превращает изображение в вектор, в котором соседние пиксели (100, 200) и (101, 200) становятся просто соседними числами без пространственной связи.

**Проблема 3. Нет инвариантности к сдвигу.** Сдвиг объекта на несколько пикселей для MLP — это совершенно новые данные.

### Свёртка

**Свёртка** — операция, при которой фильтр (ядро) скользит по изображению и вычисляет взвешенную сумму в каждой позиции.

**Пример фильтра 3×3** (детектор горизонтальной границы):

```
┌───────────┐
│ -1 -1 -1  │
│  0  0  0  │
│  1  1  1  │
└───────────┘
```

Применяя фильтр к фрагменту, получаем одно число — ответ фильтра в данной позиции.

**Три ключевых свойства свёртки:**

| Свойство | Что означает |
|---|---|
| Локальность | Фильтр смотрит только на маленькое окно (3×3) |
| Разделение весов | Один фильтр применяется ко всему изображению |
| Инвариантность к сдвигу | Объект сдвинулся → фильтр всё равно его найдёт |

### Padding, Stride, Pooling

**Padding** — добавление рамки из нулей, чтобы сохранить размер карты признаков.

**Stride** — шаг, с которым фильтр смещается по изображению. Stride=1 сохраняет размер, stride=2 уменьшает в 2 раза.

**Pooling** — уменьшение карты признаков:
- **Max Pooling** — берёт максимум в окне;
- **Average Pooling** — берёт среднее.

Pooling даёт инвариантность к малым сдвигам, уменьшает вычисления, увеличивает рецептивное поле.

### Архитектура CNN

Типичный блок:

```
Conv → BatchNorm → ReLU → Pool
```

Полная архитектура:

```
Вход → [Conv → BN → ReLU → Pool] × N → AdaptiveAvgPool → FC → выход
```

### Transfer Learning

Вместо обучения с нуля используем модель, предобученную на ImageNet (1.2 млн изображений, 1000 классов). Первые слои уже выучили полезные признаки (края, текстуры, формы). Мы заменяем последний слой и дообучаем модель на своей задаче.

```python
model = models.resnet18(weights='IMAGENET1K_V1')
model.fc = nn.Linear(512, 1)
```

### Data Augmentation

Создание новых примеров путём случайных преобразований: повороты, отражения, кропы, изменение яркости.

**Правило:** преобразование должно сохранять семантику. Для фотографий кошек и собак безопасны повороты на небольшие углы, отражения, изменение яркости. Отражение по горизонтали для животных допустимо — кошка остаётся кошкой. А вот для цифр или букв отражение опасно («6» превращается в «9»).

---

## Практический кейс

**Задача.** Построить систему, которая по фотографии определяет, изображена на ней **кошка** или **собака**.

**Зачем это нужно.** Бинарная классификация изображений — базовая задача компьютерного зрения. Такие системы используются в:
- **Фотосервисах** — автоматическая сортировка фотографий по питомцам.
- **Приютах для животных** — каталогизация и поиск.
- **Умных камерах** — наблюдение за домашними животными.
- **Образовании** — обучение студентов основам CV.

**Данные.** Датасет **CIFAR-10**, отфильтрованный по двум классам: `cat` (кошки) и `dog` (собаки). Датасет содержит 60 000 цветных изображений 32×32 пикселя. Мы отберём из них только кошек и собак.

**Почему этот датасет удобен:**
- **Размер:** ~163 МБ — легко скачать даже в университетской аудитории.
- **Сложность:** задача достаточно сложная, чтобы показать преимущества CNN перед MLP.
- **Наглядность:** кошки и собаки — понятные классы, результат обучения легко интерпретировать.
- **Стандартность:** CIFAR-10 — один из самых популярных бенчмарков в глубоком обучении.

**Требования:**
- Точность на валидации ≥ 75%.
- Скорость инференса ≤ 0.1 секунды на изображение.
- Объяснимость (Grad-CAM).

---

## Подготовка данных

Датасет CIFAR-10 встроен в `torchvision.datasets`, поэтому его загрузка не требует ручного скачивания — модуль сделает всё автоматически.

```python
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import datasets, transforms
```

### Шаг 1. Трансформации

```python
train_transform = transforms.Compose([
    transforms.Resize(64),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

val_transform = transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
```

### Шаг 2. Фильтрация классов

В CIFAR-10 класс `cat` имеет индекс 3, класс `dog` — индекс 5. Создадим обёртку `Dataset`, которая оставит только эти два класса и преобразует метки в бинарные (0 — кошка, 1 — собака).

```python
CAT_IDX, DOG_IDX = 3, 5

class BinaryCIFAR(Dataset):
    """Оставляет только кошек и собак, метки — 0 (cat) и 1 (dog)."""

    def __init__(self, dataset, transform):
        self.transform = transform
        self.images = []
        self.labels = []

        for img, label in dataset:
            if label == CAT_IDX:
                self.images.append(img)
                self.labels.append(0)
            elif label == DOG_IDX:
                self.images.append(img)
                self.labels.append(1)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        return self.transform(img), label


# Загружаем полный CIFAR-10
train_full = datasets.CIFAR10(root='./data', train=True, download=True)
val_full = datasets.CIFAR10(root='./data', train=False, download=True)

# Создаём бинарные датасеты
train_dataset = BinaryCIFAR(train_full, transform=train_transform)
val_dataset = BinaryCIFAR(val_full, transform=val_transform)

print(f"Train size: {len(train_dataset)}")
print(f"Val size: {len(val_dataset)}")
print(f"Класс 0: кошка, Класс 1: собака")
```

### Шаг 3. DataLoader

```python
from torch.utils.data import DataLoader

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=2)

print(f"Train batches: {len(train_loader)}")
print(f"Val batches: {len(val_loader)}")
```

**Задание 1.1.** Выведите 9 случайных изображений в сетке 3×3 с указанием классов (cat/dog). Визуально определите, какие признаки отличают кошку от собаки: форма ушей, морды, текстура шерсти, цвет.

```python
import matplotlib.pyplot as plt

def denormalize(img):
    """Преобразует тензор [3, H, W] обратно в изображение для отображения."""
    img = img.permute(1, 2, 0).numpy()
    img = img * 0.5 + 0.5
    return np.clip(img, 0, 1)

fig, axes = plt.subplots(3, 3, figsize=(9, 9))
class_names = ['cat', 'dog']

for ax in axes.flat:
    idx = np.random.randint(len(train_dataset))
    img, label = train_dataset[idx]
    ax.imshow(denormalize(img))
    ax.set_title(class_names[label])
    ax.axis('off')

plt.suptitle('Примеры из датасета CIFAR-10 (cat vs dog)', fontsize=14)
plt.tight_layout()
plt.show()
```

---

## Ход работы

### 1. Импорты и настройка

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import models

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             ConfusionMatrixDisplay)
import time
from copy import deepcopy
from PIL import Image

torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"PyTorch version: {torch.__version__}")
```

### 2. Архитектура SimpleCNN

Реализуйте CNN с нуля. Архитектура адаптирована под изображения 64×64.

```python
def conv_block(in_channels, out_channels):
    """Conv → BatchNorm → ReLU → Pool."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(3, 32),      # [3, 64, 64] → [32, 32, 32]
            conv_block(32, 64),     # → [64, 16, 16]
            conv_block(64, 128),    # → [128, 8, 8]
            conv_block(128, 256),   # → [256, 4, 4]
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


model = SimpleCNN(num_classes=1).to(device)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable parameters: {total_params:,}")
```

**Задание 2.1.** Сколько параметров в первом и втором свёрточных слоях? Как это число соотносится с общим числом параметров модели? Сравните с MLP, у которого первый слой имел бы такое же число входов.

### 3. Функция потерь и оптимизатор

Для бинарной классификации используем `BCEWithLogitsLoss` — она объединяет сигмоиду и бинарную кросс-энтропию, и численно стабильна.

```python
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)
```

### 4. Функции обучения и оценки

```python
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.float().unsqueeze(1).to(device)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * X_batch.size(0)
        preds = (torch.sigmoid(logits) > 0.5).float()
        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_probs, all_labels = [], []
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.float().unsqueeze(1).to(device)

        logits = model(X_batch)
        loss = criterion(logits, y_batch)

        running_loss += loss.item() * X_batch.size(0)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)

        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())

    return running_loss / total, correct / total, np.array(all_probs), np.array(all_labels)


def fit(model, train_loader, val_loader, criterion, optimizer, scheduler,
        epochs=15, patience=5, device=device, verbose=True):
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss = float("inf")
    best_weights = deepcopy(model.state_dict())
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()
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
```

### 5. Обучение SimpleCNN

```python
history = fit(model, train_loader, val_loader, criterion, optimizer, scheduler,
              epochs=15, patience=5)
plot_history(history, "SimpleCNN")
```

**Задание 5.1.** Наблюдаете ли вы переобучение? В какой момент train accuracy значительно опережает val accuracy? Какую роль играет Dropout(0.5) в классификаторе?

**Задание 5.2.** Сохраните лучшие веса модели:

```python
torch.save(model.state_dict(), 'best_simple_cnn.pt')
```

### 6. Transfer Learning с ResNet-18

```python
resnet = models.resnet18(weights='IMAGENET1K_V1')
resnet.fc = nn.Linear(512, 1)
resnet = resnet.to(device)

total_params = sum(p.numel() for p in resnet.parameters())
print(f"ResNet-18 parameters: {total_params:,}")
```

**Стратегия A. Feature Extraction** — замораживаем всё, кроме последнего слоя:

```python
for param in resnet.parameters():
    param.requires_grad = False
for param in resnet.fc.parameters():
    param.requires_grad = True
```

**Стратегия B. Fine-tuning** — размораживаем все параметры:

```python
for param in resnet.parameters():
    param.requires_grad = True
```

Обучение (fine-tuning, lr=1e-4):

```python
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(resnet.parameters(), lr=1e-4, weight_decay=1e-2)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

history_resnet = fit(resnet, train_loader, val_loader, criterion, optimizer,
                     scheduler, epochs=10, patience=5)
plot_history(history_resnet, "ResNet-18")
```

**Задание 6.1.** Заполните таблицу:

| Модель | Параметры | Best Val Acc | Время обучения (с) |
|---|---|---|---|
| SimpleCNN | | | |
| ResNet-18 (fine-tuning) | | | |

**Задание 6.2.** Напишите 5–7 предложений: какая модель показала лучший результат? Почему transfer learning работает лучше, чем обучение с нуля на небольшом датасете? Приведите не менее двух причин.

### 7. Визуализация фильтров первого слоя

```python
weights = model.features[0][0].weight.data.cpu()  # [32, 3, 3, 3]

fig, axes = plt.subplots(4, 8, figsize=(16, 8))
for i, ax in enumerate(axes.flat):
    if i < weights.shape[0]:
        w = weights[i].permute(1, 2, 0).numpy()
        w = (w - w.min()) / (w.max() - w.min() + 1e-8)
        ax.imshow(w)
        ax.set_title(f'F{i}', fontsize=8)
    ax.axis('off')
plt.suptitle('Фильтры первого слоя SimpleCNN')
plt.tight_layout()
plt.show()
```

**Задание 7.1.** Опишите, что вы видите на фильтрах. Есть ли среди них детекторы краёв, цветовых переходов, текстур? Чем они похожи на фильтры, которые мог бы использовать человек?

### 8. Визуализация feature maps

```python
activations = {}

def get_activation(name):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook

model.features[0][0].register_forward_hook(get_activation('conv1'))

model.eval()
img, label = val_dataset[0]
with torch.no_grad():
    _ = model(img.unsqueeze(0).to(device))

act = activations['conv1'][0].cpu()  # [32, H, W]

fig, axes = plt.subplots(4, 4, figsize=(12, 12))
for i, ax in enumerate(axes.flat):
    ax.imshow(act[i].numpy(), cmap='viridis')
    ax.set_title(f'Feature map {i}')
    ax.axis('off')
plt.suptitle(f'Feature maps (класс: {"dog" if label else "cat"})')
plt.tight_layout()
plt.show()
```

**Задание 8.1.** Опишите, что вы видите на feature maps. Какие области изображения активируют фильтры сильнее? Совпадает ли это с интуитивным представлением о признаках кошки/собаки?

### 9. Grad-CAM: где модель «смотрит»

Grad-CAM показывает, какие области изображения повлияли на решение модели.

```python
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate(self, input_tensor):
        self.model.eval()
        output = self.model(input_tensor)
        self.model.zero_grad()
        output[0, 0].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam[0, 0].cpu().numpy()


target_layer = model.features[3][0]
grad_cam = GradCAM(model, target_layer)

# Применим к трём случайным изображениям
fig, axes = plt.subplots(3, 3, figsize=(12, 12))
import matplotlib.cm as cm

for row in range(3):
    img, label = val_dataset[np.random.randint(len(val_dataset))]
    input_tensor = img.unsqueeze(0).to(device)
    cam = grad_cam.generate(input_tensor)

    img_show = denormalize(img)

    axes[row, 0].imshow(img_show)
    axes[row, 0].set_title(f'Оригинал ({"dog" if label else "cat"})')
    axes[row, 0].axis('off')

    axes[row, 1].imshow(cam, cmap='jet')
    axes[row, 1].set_title('Grad-CAM')
    axes[row, 1].axis('off')

    cam_resized = np.array(Image.fromarray(cam).resize(img_show.shape[:2][::-1]))
    heatmap = cm.jet(cam_resized)[:, :, :3]
    overlay = np.clip(0.5 * img_show + 0.5 * heatmap, 0, 1)
    axes[row, 2].imshow(overlay)
    axes[row, 2].set_title('Наложение')
    axes[row, 2].axis('off')

plt.suptitle('Grad-CAM: где модель видит кошку/собаку', fontsize=14)
plt.tight_layout()
plt.show()
```

**Задание 9.1.** Для 5 изображений постройте Grad-CAM. Смотрит ли модель на морду, уши, текстуру шерсти — или на фон? Согласуется ли это с тем, как человек отличает кошку от собаки?

### 10. Data Augmentation — визуализация

```python
# Возьмём одно изображение из CIFAR-10
img_pil, label = datasets.CIFAR10(root='./data', train=True, download=True)[0]

augmentations = [
    ('Оригинал', transforms.Compose([transforms.Resize(64), transforms.ToTensor()])),
    ('Поворот +15°', transforms.Compose([transforms.Resize(64), transforms.RandomRotation((15, 15)), transforms.ToTensor()])),
    ('Поворот -15°', transforms.Compose([transforms.Resize(64), transforms.RandomRotation((-15, -15)), transforms.ToTensor()])),
    ('Отражение', transforms.Compose([transforms.Resize(64), transforms.RandomHorizontalFlip(p=1.0), transforms.ToTensor()])),
    ('Яркость +', transforms.Compose([transforms.Resize(64), transforms.ToTensor(), transforms.Lambda(lambda x: x + 0.3)])),
    ('Яркость -', transforms.Compose([transforms.Resize(64), transforms.ToTensor(), transforms.Lambda(lambda x: x - 0.3)])),
]

fig, axes = plt.subplots(1, 6, figsize=(18, 4))
for i, (name, aug) in enumerate(augmentations):
    img_aug = aug(img_pil)
    axes[i].imshow(img_aug.permute(1, 2, 0).numpy())
    axes[i].set_title(name, fontsize=10)
    axes[i].axis('off')
plt.suptitle('Data Augmentation: одно изображение — разные варианты', fontsize=14)
plt.tight_layout()
plt.show()
```

**Задание 10.1.** Какие преобразования безопасны для кошек и собак, а какие могут изменить смысл? Почему отражение по горизонтали допустимо для животных, но опасно для цифр и букв?

### 11. Финальная оценка лучшей модели

Выберите лучшую модель (SimpleCNN или ResNet-18). Оцените её на валидационной выборке.

```python
# Загружаем лучшую модель (пример для SimpleCNN)
model.load_state_dict(torch.load('best_simple_cnn.pt'))
model.eval()

test_loss, test_acc, y_prob, y_true = evaluate(model, val_loader, criterion, device)
y_pred = (y_prob > 0.5).astype(int)

print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
print(f"Precision: {precision_score(y_true, y_pred):.4f}")
print(f"Recall:    {recall_score(y_true, y_pred):.4f}")
print(f"F1:        {f1_score(y_true, y_pred):.4f}")
print(f"AUC:       {roc_auc_score(y_true, y_prob):.4f}")

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['cat', 'dog'])
fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(ax=ax, cmap="Blues")
plt.title("Confusion Matrix — Best CNN")
plt.tight_layout()
plt.show()
```

**Задание 11.1.** Постройте confusion matrix. Каких животных модель путает чаще — кошек с собаками или собак с кошками? Как вы думаете, почему?

**Задание 11.2.** Ответьте письменно: почему для задачи «кошка vs собака» accuracy — не единственная важная метрика? В каких случаях precision важнее recall, а в каких — наоборот?

### 12. Примеры ошибок модели

```python
# Найдём примеры, где модель ошиблась
errors_idx = np.where(y_pred.flatten() != y_true.flatten())[0][:6]

fig, axes = plt.subplots(2, 3, figsize=(12, 8))
class_names = ['cat', 'dog']

for ax, idx in zip(axes.flat, errors_idx):
    img, _ = val_dataset[idx]
    ax.imshow(denormalize(img))
    true_label = class_names[int(y_true[idx])]
    pred_label = class_names[int(y_pred[idx])]
    prob = y_prob[idx][0]
    ax.set_title(f'Истина: {true_label}\nПредсказано: {pred_label} ({prob:.2f})')
    ax.axis('off')

plt.suptitle('Примеры ошибок модели', fontsize=14)
plt.tight_layout()
plt.show()
```

**Задание 12.1.** Посмотрите на примеры ошибок. Почему модель могла ошибиться? Согласны ли вы с её решением? Есть ли у этих изображений что-то общее?

---

## Содержание отчёта

Отчёт оформляется в **PDF** или **Word** и содержит:

1. **Титульный лист** — дисциплина, номер работы, ФИО, группа, дата.
2. **Цель работы** — своими словами.
3. **Описание датасета** — CIFAR-10 (бинарная задача cat vs dog): источник, число изображений, размер, примеры визуализации.
4. **Описание архитектуры SimpleCNN** — число слоёв, фильтров, наличие BatchNorm/Dropout, число параметров, блок-схема forward pass.
5. **Эксперименты** — сравнение SimpleCNN и ResNet-18 (таблица), влияние transfer learning.
6. **Результаты лучшей модели** — accuracy, precision, recall, F1, AUC, confusion matrix.
7. **Визуализации** — фильтры первого слоя, feature maps, Grad-CAM, примеры ошибок.
8. **Графики обучения** — loss и accuracy для лучшей конфигурации.
9. **Выводы** — 3–5 пунктов: какая модель показала лучший результат, что было неожиданным, что бы вы улучшили.
10. **Ответы на контрольные вопросы** — см. ниже.
11. **Приложение** — ссылка на Colab-notebook или листинг ключевых функций.

**Объём:** 6–10 страниц без приложения.

---

## Контрольные вопросы

1. Почему MLP плохо работает с изображениями? Назовите три проблемы.
2. Что такое свёртка? Как фильтр 3×3 превращает фрагмент изображения в одно число?
3. Что такое feature map? Что означает большое положительное число в feature map?
4. Зачем нужен padding? Что произойдёт с изображением без padding после 128 свёрточных слоёв?
5. Чем Max Pooling отличается от Average Pooling? Почему для различения кошек и собак лучше Max Pooling?
6. Что такое stride? Как stride влияет на размер выхода?
7. Почему ResNet использует skip connections? Какую проблему они решают?
8. Что такое transfer learning? Почему он работает лучше, чем обучение с нуля на небольшом датасете?
9. Какие преобразования безопасны для кошек и собак, а какие — опасны? Почему?
10. Что показывает Grad-CAM? Почему это важно для интерпретации модели?
11. Почему `BCEWithLogitsLoss` предпочтительнее, чем `BCELoss` + `Sigmoid` в модели?
12. Что произойдёт, если забыть вызвать `model.eval()` перед валидацией?
13. Вы обучили CNN на CIFAR-10 (cat vs dog) с accuracy 85%. Коллега предлагает увеличить число фильтров в 2 раза. Какие риски это несёт? Как бы вы проверили, улучшит ли это модель?
14. В каких практических задачах важнее recall, а в каких — precision? Приведите примеры для задачи классификации животных.

---

## Критерии оценивания

| Критерий | Баллы |
|---|---|
| Корректная загрузка и подготовка данных CIFAR-10 | 1 |
| Реализация и обучение SimpleCNN | 2 |
| Применение transfer learning (ResNet-18) | 2 |
| Визуализация фильтров и feature maps | 1 |
| Grad-CAM и анализ «внимания» модели | 1 |
| Оценка метрик и confusion matrix | 1 |
| Качество отчёта и выводов | 1 |
| Ответы на контрольные вопросы | 1 |
| **Итого** | **10** |

---

## Полезные ссылки

- [CIFAR-10 — официальный сайт](https://www.cs.toronto.edu/~kriz/cifar.html)
- [PyTorch CNN tutorial](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
- [ResNet paper](https://arxiv.org/abs/1512.03385)
- [Grad-CAM paper](https://arxiv.org/abs/1610.02391)
- [torchvision models](https://pytorch.org/vision/stable/models.html)

---

## Альтернативные датасеты

Если CIFAR-10 покажется слишком сложным, можно рассмотреть более простые варианты:

| Датасет | Размер | Особенность |
|---|---|---|
| **Fashion-MNIST** | ~30 МБ | 10 классов одежды, серые изображения 28×28 |
| **Cats vs Dogs (tiny)** | ~10 МБ | 275 тренировочных изображений, очень маленький |
| **Oxford-IIIT Pet** | ~800 МБ | 37 пород кошек и собак |
| **Imagenette** | ~1.5 ГБ | 10 классов из ImageNet |

**Рекомендация:** CIFAR-10 — оптимальный баланс между размером (~163 МБ), сложностью и наглядностью.