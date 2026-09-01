# Машинное обучение и анализ данных. Часть 2

Второй семестр дисциплины полностью посвящён **нейронным сетям** — от базовых понятий до современных архитектур и прикладного использования.

**Предварительные требования (семестр 1):** регрессия, классификация, кластеризация, бутстреп, методы понижения размерности данных.

**Формат:** 8 лекционных пар (по 2 академических часа) + 8 лабораторных работ.

**Основные инструменты:** Python, NumPy, PyTorch, Hugging Face, Google Colab.

---

## Лабораторные работы

| № | Название | Файл |
|---|----------|------|
| 1 | MLP «с нуля» на NumPy — предсказание риска ИБС | [lab1.md](lab1.md) |
| 2 | Обучение MLP в PyTorch | lab2.md |
| 3 | CNN для классификации изображений | lab3.md |
| 4 | Transfer Learning на своём датасете | lab4.md |
| 5 | LSTM для текста или временного ряда | lab5.md |
| 6 | Transformer / предобученная модель | lab6.md |
| 7 | Генеративная модель (VAE / DCGAN) | lab7.md |
| 8 | Деплой модели (мини-MLOps) | lab8.md |

---

## План лекций и лабораторных работ

### Лекция 1. Введение в нейронные сети: от перцептрона к MLP

**Содержание:**
- Мотивация: ограничения линейных моделей
- Искусственный нейрон, функции активации (sigmoid, ReLU, softmax)
- Многослойный перceptron (MLP), функции потерь
- Алгоритм обратного распространения ошибки (backpropagation)
- Проблемы обучения: vanishing gradients, переобучение

**Лабораторная 1.** MLP «с нуля» на NumPy — предсказание риска ишемической болезни сердца (Cleveland Heart Disease Dataset). Реализация forward/backward pass, сравнение с логистической регрессией.

---

### Лекция 2. Обучение глубоких сетей: оптимизация, регуляризация, фреймворки

**Содержание:**
- SGD, mini-batch, оптимизаторы (Momentum, Adam, AdamW)
- Регуляризация: L2, dropout, early stopping, data augmentation
- BatchNorm, learning rate scheduling, gradient clipping
- PyTorch: Tensor, autograd, Dataset/DataLoader, цикл обучения
- Типичные ошибки при обучении

**Лабораторная 2.** Обучение MLP в PyTorch — классификация изображений (Fashion-MNIST / CIFAR-10). Сравнение оптимизаторов, dropout, BatchNorm, логирование метрик.

---

### Лекция 3. Сверточные нейронные сети (CNN)

**Содержание:**
- Идея свёртки, padding, stride, pooling
- Архитектура CNN: Conv → BN → ReLU → Pool → FC
- Классические архитектуры: LeNet, AlexNet, VGG, ResNet
- Визуализация фильтров и feature maps
- Data augmentation для изображений

**Лабораторная 3.** CNN для классификации изображений (CIFAR-10). Augmentation, сравнение с MLP на «развёрнутых» пикселях, визуализация фильтров.

---

### Лекция 4. Transfer Learning и прикладное компьютерное зрение

**Содержание:**
- Предобучение на ImageNet, fine-tuning vs feature extraction
- Работа с малыми датасетами
- Метрики для несбалансированных классов
- Grad-CAM и интерпретируемость CNN
- Обзор detection и segmentation

**Лабораторная 4.** Transfer Learning — fine-tune ResNet-18 на прикладном датасете. Confusion matrix, Grad-CAM, анализ ошибок.

---

### Лекция 5. Рекуррентные сети и модели последовательностей

**Содержание:**
- RNN, проблема длинных зависимостей
- LSTM и GRU
- Word embeddings (Word2Vec, GloVe)
- Задачи: классификация текста, прогноз временных рядов
- Ограничения RNN → мост к attention

**Лабораторная 5.** LSTM для текста (sentiment analysis) или временного ряда. Сравнение с baseline (TF-IDF + Logistic Regression / ARIMA).

---

### Лекция 6. Механизм внимания и Transformer

**Содержание:**
- Self-attention: Query, Key, Value
- Multi-head attention, positional encoding
- Архитектура Transformer
- BERT vs GPT, Vision Transformer (ViT)
- Экосистема Hugging Face

**Лабораторная 6.** Fine-tuning предобученной модели (DistilBERT) для классификации текста. Сравнение с LSTM из лабораторной 5.

---

### Лекция 7. Генеративные модели: AE, VAE, GAN, diffusion

**Содержание:**
- Autoencoder и связь с PCA
- Variational Autoencoder (VAE)
- Generative Adversarial Networks (GAN)
- Diffusion models (обзор)
- Multimodal models (CLIP — концептуально)

**Лабораторная 7.** VAE или DCGAN на MNIST / Fashion-MNIST. Генерация образцов, визуализация latent space (t-SNE / UMAP).

---

### Лекция 8. Продвинутые темы и итоги: от модели к продукту

**Содержание:**
- Self-supervised learning, Graph Neural Networks (обзор)
- Reinforcement learning + deep RL (обзор)
- Оптимизация inference: quantization, pruning, distillation
- MLOps: сохранение модели, REST API, Docker (обзор)
- Responsible AI, итоги курса

**Лабораторная 8.** Деплой модели — сохранение weights, REST API на FastAPI, инструкция по запуску.

---

## Календарная схема (16 недель)

| Недели | Лекция | Лабораторная |
|--------|--------|--------------|
| 1–2 | Лек. 1 — Введение в нейросети, MLP, backprop | Лаб. 1 — MLP на NumPy |
| 3–4 | Лек. 2 — Оптимизация, регуляризация, PyTorch | Лаб. 2 — MLP в PyTorch |
| 5–6 | Лек. 3 — Сверточные нейронные сети | Лаб. 3 — CNN |
| 7–8 | Лек. 4 — Transfer Learning | Лаб. 4 — Fine-tuning |
| 9–10 | Лек. 5 — RNN, LSTM | Лаб. 5 — LSTM |
| 11–12 | Лек. 6 — Attention, Transformer | Лаб. 6 — Transformer |
| 13–14 | Лек. 7 — Генеративные модели | Лаб. 7 — VAE / GAN |
| 15–16 | Лек. 8 — Продвинутые темы, MLOps | Лаб. 8 — Деплой |

---


## Рекомендуемая литература

1. **Goodfellow, Bengio, Courville** — *Deep Learning* (теория)
2. **Raschka, Liu, Mirjalili** — *Machine Learning with PyTorch and Scikit-Learn*
3. **[d2l.ai](https://d2l.ai/)** — интерактивный учебник (бесплатно)
4. **[fast.ai](https://www.fast.ai/)** — практический курс
5. **[Hugging Face Course](https://huggingface.co/learn)** — для работы с Transformer
