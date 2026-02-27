# predict.py
import cv2
import numpy as np
from tensorflow.keras.models import load_model


def predict_mask(model, image_path, img_size=(256, 256)):
    # Загрузка и предобработка изображения
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    original_shape = img.shape[:2]
    img_resized = cv2.resize(img, img_size)
    img_norm = img_resized / 255.0
    img_batch = np.expand_dims(img_norm, axis=0)

    # Предсказание
    mask = model.predict(img_batch)[0]  # (256,256,1)
    mask = (mask > 0.5).astype(np.uint8) * 255

    # Восстанавливаем исходный размер
    mask_full = cv2.resize(mask, (original_shape[1], original_shape[0]))
    return mask_full


def calculate_area(mask, pixel_to_cm_ratio=None):
    # Площадь в пикселях
    pixel_area = np.sum(mask > 0)
    print(f'Площадь в пикселях: {pixel_area}')

    # Если известен масштаб (например, сколько пикселей в 1 см), переводим в см²
    if pixel_to_cm_ratio:
        real_area = pixel_area / (pixel_to_cm_ratio ** 2)
        print(f'Площадь в см²: {real_area:.2f}')
        return real_area
    return pixel_area


if __name__ == '__main__':
    model = load_model('best_model.h5')
    mask = predict_mask(model, 'test_photo.jpg')
    cv2.imwrite('predicted_mask.png', mask)
    area = calculate_area(mask)  # если нет масштаба, просто пиксели
    print(f'Площадь ростка: {area} пикселей')