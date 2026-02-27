# data_loader.py
# Скрипт для загрузки данных и аугментации изображений для сегментации

import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import albumentations as A  # альтернативная библиотека для аугментации (лучше для сегментации)


def load_data(image_dir, mask_dir, img_size=(256, 256)):
    """
    Загружает изображения и соответствующие маски из папок.

    Параметры:
        image_dir (str): путь к папке с изображениями
        mask_dir (str): путь к папке с масками
        img_size (tuple): размер, до которого нужно изменить изображения (width, height)

    Возвращает:
        tuple: (массив изображений, массив масок) нормализованные и подготовленные для обучения
    """
    images = []
    masks = []

    # Получаем список всех файлов в папке с изображениями
    for filename in os.listdir(image_dir):
        # Проверяем, что файл - изображение
        if filename.endswith(('.jpg', '.png', '.jpeg', '.JPG', '.PNG')):

            # Формируем пути к изображению и маске
            img_path = os.path.join(image_dir, filename)

            # Предполагаем, что маска имеет то же имя, но расширение .png
            # Если у вас другое расширение для масок, измените здесь
            mask_filename = filename.replace('.jpg', '.png').replace('.jpeg', '.png').replace('.JPG', '.png')
            mask_path = os.path.join(mask_dir, mask_filename)

            # Проверяем, существует ли файл маски
            if not os.path.exists(mask_path):
                print(f"Предупреждение: маска для {filename} не найдена. Пропускаем.")
                continue

            # Загружаем изображение
            img = cv2.imread(img_path)
            if img is None:
                print(f"Ошибка: не удалось загрузить {img_path}")
                continue

            # Конвертируем BGR в RGB (OpenCV загружает в BGR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Изменяем размер изображения
            img = cv2.resize(img, img_size)

            # Загружаем маску (в оттенках серого)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"Ошибка: не удалось загрузить маску {mask_path}")
                continue

            # Изменяем размер маски
            mask = cv2.resize(mask, img_size)

            # Бинаризуем маску (растение = 1, фон = 0)
            # Предполагаем, что на маске растение белое (255), фон черный (0)
            mask = (mask > 127).astype(np.float32)

            images.append(img)
            masks.append(mask)

    print(f"Загружено {len(images)} изображений и масок")

    # Преобразуем в numpy массивы и нормализуем изображения
    X = np.array(images, dtype=np.float32) / 255.0
    y = np.array(masks, dtype=np.float32).reshape(-1, img_size[0], img_size[1], 1)

    return X, y


# ==================== МЕТОД 1: Аугментация с Keras ====================

def create_keras_augmentor():
    """
    Создает аугментатор с использованием Keras ImageDataGenerator.

    Внимание: для сегментации нужно одинаково трансформировать и изображение, и маску.
    Этот метод требует специального генератора (см. train_generator в train.py).
    """
    return ImageDataGenerator(
        rotation_range=30,  # поворот до 30 градусов
        width_shift_range=0.1,  # горизонтальный сдвиг до 10%
        height_shift_range=0.1,  # вертикальный сдвиг до 10%
        shear_range=0.1,  # сдвиг
        zoom_range=0.2,  # масштабирование до 20%
        horizontal_flip=True,  # горизонтальное отражение
        vertical_flip=False,  # вертикальное отражение (для растений обычно не нужно)
        fill_mode='nearest',  # как заполнять пустые области
        brightness_range=[0.8, 1.2]  # изменение яркости
    )


def keras_augment(img, mask, augmentor):
    """
    Применяет аугментацию Keras к паре изображение-маска.

    Параметры:
        img: изображение (H, W, 3)
        mask: маска (H, W, 1)
        augmentor: настроенный ImageDataGenerator

    Возвращает:
        tuple: (аугментированное изображение, аугментированная маска)
    """
    # Объединяем изображение и маску для одинаковых трансформаций
    combined = np.concatenate([img, mask], axis=-1)  # (H, W, 4)

    # Применяем трансформацию
    transformed = augmentor.random_transform(combined)

    # Разделяем обратно
    aug_img = transformed[..., :3]
    aug_mask = transformed[..., 3:]

    return aug_img, aug_mask


# ==================== МЕТОД 2: Аугментация с Albumentations (РЕКОМЕНДУЕТСЯ) ====================

def create_albumentations_augmentor():
    """
    Создает аугментатор с использованием библиотеки Albumentations.
    Эта библиотека специально разработана для задач сегментации и
    автоматически применяет одинаковые трансформации к изображению и маске.

    Установка: pip install albumentations
    """
    return A.Compose([
        A.RandomRotate90(p=0.5),
        A.Flip(p=0.5),
        A.Transpose(p=0.5),
        A.OneOf([
            A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
            A.GridDistortion(p=0.5),
            A.OpticalDistortion(distort_limit=2, shift_limit=0.5, p=0.5),
        ], p=0.3),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.5),
        A.OneOf([
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        ], p=0.5),
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
            A.GaussianBlur(blur_limit=(3, 7), p=0.5),
            A.MotionBlur(blur_limit=(3, 7), p=0.5),
        ], p=0.3),
    ])


class AlbumentationsGenerator:
    """
    Генератор данных с аугментацией через Albumentations.
    Используйте этот класс в train.py для обучения.
    """

    def __init__(self, X, y, batch_size=8, augment=True):
        """
        Параметры:
            X: массив изображений
            y: массив масок
            batch_size: размер батча
            augment: применять ли аугментацию
        """
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.augment = augment
        self.augmentor = create_albumentations_augmentor() if augment else None
        self.on_epoch_end()

    def __len__(self):
        """Количество батчей за эпоху"""
        return int(np.ceil(len(self.X) / self.batch_size))

    def on_epoch_end(self):
        """Перемешиваем данные в конце каждой эпохи"""
        self.indices = np.arange(len(self.X))
        np.random.shuffle(self.indices)

    def __getitem__(self, index):
        """Генерируем один батч"""
        # Получаем индексы для текущего батча
        start = index * self.batch_size
        end = min(start + self.batch_size, len(self.X))
        batch_indices = self.indices[start:end]

        # Создаем батч
        batch_X = self.X[batch_indices].copy()
        batch_y = self.y[batch_indices].copy()

        # Применяем аугментацию, если нужно
        if self.augment:
            for i in range(len(batch_X)):
                augmented = self.augmentor(image=batch_X[i], mask=batch_y[i])
                batch_X[i] = augmented['image']
                batch_y[i] = np.expand_dims(augmented['mask'], axis=-1)

        return batch_X, batch_y


# ==================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ====================

def split_data(X, y, validation_split=0.2):
    """
    Разделяет данные на обучающую и валидационную выборки.

    Параметры:
        X: массив изображений
        y: массив масок
        validation_split: доля данных для валидации

    Возвращает:
        tuple: (X_train, X_val, y_train, y_val)
    """
    split_idx = int(len(X) * (1 - validation_split))

    # Перемешиваем данные перед разделением
    indices = np.random.permutation(len(X))
    X_shuffled = X[indices]
    y_shuffled = y[indices]

    X_train = X_shuffled[:split_idx]
    X_val = X_shuffled[split_idx:]
    y_train = y_shuffled[:split_idx]
    y_val = y_shuffled[split_idx:]

    print(f"Обучающая выборка: {len(X_train)} изображений")
    print(f"Валидационная выборка: {len(X_val)} изображений")

    return X_train, X_val, y_train, y_val


def visualize_augmentation(X, y, num_examples=3):
    """
    Визуализирует примеры аугментации для проверки.
    """
    import matplotlib.pyplot as plt

    augmentor = create_albumentations_augmentor()

    fig, axes = plt.subplots(num_examples, 4, figsize=(12, 3 * num_examples))

    for i in range(num_examples):
        idx = np.random.randint(len(X))

        # Оригинал
        axes[i, 0].imshow(X[idx])
        axes[i, 0].set_title('Оригинал')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(y[idx].squeeze(), cmap='gray')
        axes[i, 1].set_title('Маска')
        axes[i, 1].axis('off')

        # Аугментированная версия
        augmented = augmentor(image=X[idx], mask=y[idx])
        aug_img = augmented['image']
        aug_mask = augmented['mask']

        axes[i, 2].imshow(aug_img)
        axes[i, 2].set_title('Аугментированное')
        axes[i, 2].axis('off')

        axes[i, 3].imshow(aug_mask, cmap='gray')
        axes[i, 3].set_title('Аугментированная маска')
        axes[i, 3].axis('off')

    plt.tight_layout()
    plt.savefig('augmentation_examples.png')
    plt.show()


# ==================== ТЕСТИРОВАНИЕ ====================

if __name__ == '__main__':
    # Этот код выполняется только при прямом запуске файла
    # Используется для тестирования функций загрузки и аугментации

    print("Тестирование загрузчика данных...")

    # Проверяем, существуют ли папки с данными
    if os.path.exists('dataset/images') and os.path.exists('dataset/masks'):
        # Загружаем данные
        X, y = load_data('dataset/images', 'dataset/masks')
        print(f"Форма X: {X.shape}")
        print(f"Форма y: {y.shape}")

        # Визуализируем аугментацию
        visualize_augmentation(X, y)

        # Тестируем разделение данных
        split_data(X, y)
    else:
        print("Папки dataset/images и dataset/masks не найдены.")
        print("Создайте их и добавьте изображения для тестирования.")