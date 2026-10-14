# Лабораторная работа №3

## Свёрточные нейронные сети (CNN) для классификации изображений

**Курс:** Машинное обучение и анализ данных
**Семестр:** 2
**Тема:** Свёрточные нейронные сети
**Среда выполнения:** Google Colab (рекомендуется) / Jupyter Notebook
**Язык:** Python 3

---

## Цель работы

Изучить архитектуру свёрточных нейронных сетей (CNN), освоить практические навыки обучения CNN с нуля и с применением transfer learning, научиться визуализировать работу сети (фильтры, feature maps, Grad-CAM) и оценивать качество модели на реальной задаче классификации изображений.

---

## Задачи

1. Разобраться, почему полносвязные сети (MLP) плохо подходят для изображений.
2. Реализовать свёрточную сеть с нуля на PyTorch.
3. Обучить CNN на реальном датасете сельскохозяйственных изображений.
4. Применить transfer learning с предобученной ResNet-18.
5. Визуализировать фильтры первого слоя и карты признаков.
6. Построить тепловые карты Grad-CAM и проверить, куда «смотрит» модель.
7. Оценить качество модели с помощью метрик (accuracy, precision, recall, F1).

---

## Теоретическая справка

### Почему MLP не подходит для изображений

**Проблема 1. Взрывное число параметров.** Изображение 256×256×3 содержит 196 608 входов. Первый слой MLP с 256 нейронами требует 50 331 648 весов — это несколько гигабайт памяти только для одного слоя.

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

**Правило:** преобразование должно сохранять семантику. Для зёрен безопасен поворот на 360° (зерно может лежать как угодно). Для цифр опасен поворот на 180° («6» превратится в «9»).

---

## Практический кейс

**Задача.** Построить систему обнаружения плесени на семенах пшеницы по фотографии.

**Зачем это нужно.** Перед посевом семена проверяют на качество. Плесень снижает всхожесть и может заразить весь урожай. Сейчас это делают вручную — медленно и субъективно.

**Данные.** Датасет **GrainSpace** (подмножество пшеницы). Классы: **NORMAL** (здоровое зерно) и **MOULDY** (плесневелое).

**Требования:**
- Точность ≥ 92%
- Скорость ≤ 0.5 сек на зерно
- Объяснимость (Grad-CAM)

**Ссылка на датасет:** [github.com/hellodfan/GrainSpace](https://github.com/hellodfan/GrainSpace)

---

## Подготовка данных

Скачайте подмножество **TRAIN-Wheat-M600** с официального сайта и распакуйте так:

```
data/grainspace/
├── train/
│   ├── NORMAL/
│   └── MOULDY/
└── val/
    ├── NORMAL/
    └── MOULDY/
```

Для валидации возьмите 15–20% изображений из каждого класса.

---

## Ход работы

### 1. Импорты и настройка

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, ConfusionMatrixDisplay)
import time
from copy import deepcopy
from PIL import Image

torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"PyTorch version: {torch.__version__}")
```

### 2. Загрузка данных

```python
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomRotation(360),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder('data/grainspace/train', transform=train_transform)
val_dataset = datasets.ImageFolder('data/grainspace/val', transform=val_transform)

print(f"Train size: {len(train_dataset)}")
print(f"Val size: {len(val_dataset)}")
print(f"Classes: {train_dataset.classes}")
```

Создайте DataLoader:

```python
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2)
```

**Задание 2.1.** Выведите 9 случайных изображений в сетке 3×3 с указанием классов. Визуально определите признаки плесени.

### 3. Архитектура SimpleCNN

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
            conv_block(3, 32),      # [3, 224, 224] → [32, 112, 112]
            conv_block(32, 64),     # → [64, 56, 56]
            conv_block(64, 128),    # → [128, 28, 28]
            conv_block(128, 256),   # → [256, 14, 14]
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

**Задание 3.1.** Сколько параметров в первом и втором свёрточных слоях? Сравните с MLP из лабораторной №2.

### 4. Функция потерь и оптимизатор

```python
pos_weight = torch.tensor([3.0]).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
```

**Зачем `pos_weight=3.0`?** Здоровых зёрен в 3 раза больше, чем плесневелых. Параметр усиливает вклад плесени в loss, чтобы модель не «ленилась» предсказывать только здоровое зерно.

### 5. Функции обучения

```python
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.float().unsqueeze(1).to(device)
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
        X_batch, y_batch = X_batch.to(device), y_batch.float().unsqueeze(1).to(device)
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
        epochs=20, patience=5, device=device, verbose=True):
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

### 6. Обучение SimpleCNN

```python
history = fit(model, train_loader, val_loader, criterion, optimizer, scheduler,
              epochs=20, patience=5)
plot_history(history, "SimpleCNN")
```

**Задание 6.1.** Наблюдаете ли вы переобучение? В какой момент train accuracy значительно опережает val accuracy?

### 7. Transfer Learning с ResNet-18

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

Обучение (fine-tuning):

```python
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.0]).to(device))
optimizer = optim.AdamW(resnet.parameters(), lr=1e-4, weight_decay=1e-2)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

history_resnet = fit(resnet, train_loader, val_loader, criterion, optimizer,
                     scheduler, epochs=10, patience=5)
plot_history(history_resnet, "ResNet-18")
```

**Задание 7.1.** Заполните таблицу:

| Модель | Параметры | Best Val Acc | Время обучения (с) |
|---|---|---|---|
| SimpleCNN | | | |
| ResNet-18 | | | |

**Задание 7.2.** Напишите 5–7 предложений: какая модель показала лучший результат? Почему transfer learning работает лучше, чем обучение с нуля?

### 8. Визуализация фильтров первого слоя

```python
weights = model.features[0][0].weight.data.cpu()

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

### 9. Визуализация feature maps

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

act = activations['conv1'][0].cpu()

fig, axes = plt.subplots(4, 4, figsize=(12, 12))
for i, ax in enumerate(axes.flat):
    ax.imshow(act[i].numpy(), cmap='viridis')
    ax.set_title(f'Feature map {i}')
    ax.axis('off')
plt.suptitle(f'Feature maps (класс: {val_dataset.classes[label]})')
plt.tight_layout()
plt.show()
```

**Задание 9.1.** Опишите, что вы видите на фильтрах и feature maps. Есть ли детекторы краёв, пятен, цветовых переходов?

### 10. Grad-CAM

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

img, label = val_dataset[0]
input_tensor = img.unsqueeze(0).to(device)
cam = grad_cam.generate(input_tensor)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
img_show = img.permute(1, 2, 0).numpy()
img_show = img_show * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
img_show = np.clip(img_show, 0, 1)

axes[0].imshow(img_show); axes[0].set_title(f'Оригинал ({val_dataset.classes[label]})'); axes[0].axis('off')
axes[1].imshow(cam, cmap='jet'); axes[1].set_title('Grad-CAM'); axes[1].axis('off')

import matplotlib.cm as cm
cam_resized = np.array(Image.fromarray(cam).resize(img_show.shape[:2][::-1]))
heatmap = cm.jet(cam_resized)[:, :, :3]
overlay = np.clip(0.5 * img_show + 0.5 * heatmap, 0, 1)
axes[2].imshow(overlay); axes[2].set_title('Наложение'); axes[2].axis('off')
plt.tight_layout(); plt.show()
```

**Задание 10.1.** Для 5 изображений с плесенью постройте Grad-CAM. Смотрит ли модель на пятно плесени или на фон?

### 11. Data Augmentation — визуализация

```python
img_orig, label = datasets.ImageFolder('data/grainspace/train',
                                        transform=transforms.ToTensor())[0]

augmentations = [
    ('Оригинал', transforms.Compose([transforms.ToTensor()])),
    ('Поворот +30°', transforms.Compose([transforms.RandomRotation((30, 30)), transforms.ToTensor()])),
    ('Поворот -30°', transforms.Compose([transforms.RandomRotation((-30, -30)), transforms.ToTensor()])),
    ('Отражение', transforms.Compose([transforms.RandomHorizontalFlip(p=1.0), transforms.ToTensor()])),
    ('Яркость +', transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x + 0.3)])),
    ('Яркость -', transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x - 0.3)])),
]

fig, axes = plt.subplots(1, 6, figsize=(18, 4))
img_pil = transforms.ToPILImage()(img_orig)
for i, (name, aug) in enumerate(augmentations):
    img_aug = img_orig if name == 'Оригинал' else aug(img_pil)
    axes[i].imshow(img_aug.permute(1, 2, 0))
    axes[i].set_title(name, fontsize=10)
    axes[i].axis('off')
plt.suptitle('Data Augmentation: одно зерно — разные варианты')
plt.tight_layout(); plt.show()
```

**Задание 11.1.** Какие преобразования безопасны для зёрен, а какие могут изменить смысл? Почему поворот на 360° — хорошая идея?

### 12. Оценка лучшей модели

```python
test_loss, test_acc, y_prob, y_true = evaluate(model, val_loader, criterion, device)
y_pred = (y_prob > 0.5).astype(int)

print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
print(f"Precision: {precision_score(y_true, y_pred):.4f}")
print(f"Recall:    {recall_score(y_true, y_pred):.4f}")
print(f"F1:        {f1_score(y_true, y_pred):.4f}")

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=val_dataset.classes)
fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(ax=ax, cmap="Blues")
plt.title("Confusion Matrix — Best CNN")
plt.tight_layout(); plt.show()
```

**Задание 12.1.** Постройте confusion matrix. Какие классы модель путает чаще всего? Почему для нашей задачи recall важнее precision?

---

## Содержание отчёта

Отчёт оформляется в **PDF** или **Word** и содержит:

1. **Титульный лист** — дисциплина, номер работы, ФИО, группа, дата.
2. **Цель работы** — своими словами.
3. **Описание датасета** — GrainSpace (пшеница): источник, число изображений и классов, примеры визуализации.
4. **Описание архитектуры** — SimpleCNN: число слоёв, фильтров, наличие BatchNorm/Dropout, число параметров.
5. **Эксперименты** — сравнение SimpleCNN и ResNet-18 (таблица), влияние transfer learning.
6. **Результаты** — accuracy, precision, recall, F1, confusion matrix.
7. **Визуализации** — фильтры первого слоя, feature maps, Grad-CAM.
8. **Графики обучения** — loss и accuracy для лучшей конфигурации.
9. **Выводы** — 3–5 пунктов.
10. **Ответы на контрольные вопросы** — см. ниже.
11. **Приложение** — ссылка на Colab-notebook или листинг ключевых функций.

**Объём:** 6–10 страниц без приложения.

---

## Контрольные вопросы

1. Почему MLP плохо работает с изображениями? Назовите три проблемы.
2. Что такое свёртка? Как фильтр 3×3 превращает фрагмент изображения в одно число?
3. Что такое feature map? Что означает большое положительное число в feature map?
4. Зачем нужен padding? Что произойдёт с изображением без padding после 128 свёрточных слоёв?
5. Чем Max Pooling отличается от Average Pooling? Почему для плесени лучше Max Pooling?
6. Что такое stride? Как stride влияет на размер выхода?
7. Почему ResNet использует skip connections? Какую проблему они решают?
8. Что такое transfer learning? Почему он работает лучше, чем обучение с нуля на маленьком датасете?
9. Какие преобразования безопасны для зёрен, а какие — опасны? Почему?
10. Что показывает Grad-CAM? Почему это важно для агронома?
11. Почему `BCEWithLogitsLoss` предпочтительнее, чем `BCELoss` + `Sigmoid` в модели?
12. Что произойдёт, если забыть вызвать `model.eval()` перед валидацией?
13. Вы обучили CNN на GrainSpace с accuracy 93%. Коллега предлагает увеличить число фильтров в 2 раза. Какие риски это несёт? Как бы вы проверили, улучшит ли это модель?
14. Почему для нашей задачи recall важнее precision? Что произойдёт, если модель пропустит плесневелое зерно?

---

## Критерии оценивания

| Критерий | Баллы |
|---|---|
| Корректная загрузка и подготовка данных | 1 |
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

- [GrainSpace — официальный репозиторий](https://github.com/hellodfan/GrainSpace)
- [PyTorch CNN tutorial](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
- [ResNet paper](https://arxiv.org/abs/1512.03385)
- [Grad-CAM paper](https://arxiv.org/abs/1610.02391)
- [torchvision models](https://pytorch.org/vision/stable/models.html)