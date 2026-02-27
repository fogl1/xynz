# train.py
import numpy as np
from model import unet
from data_loader import load_data, split_data, AlbumentationsGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import os

# Параметры
IMG_SIZE = (256, 256)
BATCH_SIZE = 8
EPOCHS = 100
LEARNING_RATE = 1e-4

# Создаем папку для сохранения моделей
os.makedirs('models', exist_ok=True)


def dice_coefficient(y_true, y_pred, smooth=1e-6):
    """Метрика Dice coefficient для оценки качества сегментации"""
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)


def dice_loss(y_true, y_pred):
    """Функция потерь на основе Dice coefficient"""
    return 1 - dice_coefficient(y_true, y_pred)


def main():
    print("=" * 50)
    print("ЗАГРУЗКА ДАННЫХ")
    print("=" * 50)

    # Загрузка данных
    X, y = load_data('dataset/images', 'dataset/masks', IMG_SIZE)

    # Разделение на обучение и валидацию
    X_train, X_val, y_train, y_val = split_data(X, y, validation_split=0.2)

    print("\n" + "=" * 50)
    print("СОЗДАНИЕ МОДЕЛИ")
    print("=" * 50)

    # Создание модели
    model = unet(input_size=(*IMG_SIZE, 3))
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss=dice_loss,  # используем Dice loss вместо binary_crossentropy
        metrics=['accuracy', dice_coefficient]
    )

    # Вывод информации о модели
    model.summary()

    print("\n" + "=" * 50)
    print("НАСТРОЙКА ОБУЧЕНИЯ")
    print("=" * 50)

    # Создание генераторов с аугментацией
    train_generator = AlbumentationsGenerator(X_train, y_train, batch_size=BATCH_SIZE, augment=True)
    val_generator = AlbumentationsGenerator(X_val, y_val, batch_size=BATCH_SIZE, augment=False)

    # Callbacks для улучшения обучения
    callbacks = [
        # Сохраняем лучшую модель
        ModelCheckpoint(
            'models/best_model.h5',
            monitor='val_dice_coefficient',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        # Ранняя остановка при отсутствии прогресса
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        # Уменьшение learning rate при плато
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]

    print("\n" + "=" * 50)
    print("НАЧАЛО ОБУЧЕНИЯ")
    print("=" * 50)

    # Обучение
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )

    # Сохраняем финальную модель
    model.save('models/final_model.h5')

    print("\n" + "=" * 50)
    print("ОБУЧЕНИЕ ЗАВЕРШЕНО")
    print("=" * 50)

    # Визуализация процесса обучения
    plot_training_history(history)


def plot_training_history(history):
    """Строит графики процесса обучения"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # График потерь
    axes[0].plot(history.history['loss'], label='Train')
    axes[0].plot(history.history['val_loss'], label='Validation')
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)

    # График точности
    axes[1].plot(history.history['accuracy'], label='Train')
    axes[1].plot(history.history['val_accuracy'], label='Validation')
    axes[1].set_title('Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True)

    # График Dice coefficient
    if 'dice_coefficient' in history.history:
        axes[2].plot(history.history['dice_coefficient'], label='Train')
        axes[2].plot(history.history['val_dice_coefficient'], label='Validation')
    axes[2].set_title('Dice Coefficient')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Dice')
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig('models/training_history.png')
    plt.show()


if __name__ == '__main__':
    import tensorflow as tf

    main()