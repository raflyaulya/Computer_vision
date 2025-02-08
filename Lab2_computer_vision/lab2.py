import cv2
import numpy as np
import matplotlib.pyplot as plt

# Функция для разделения и возврата каналов RGB
def split_channels(image):
    return image[:, :, 2], image[:, :, 1], image[:, :, 0]

# Чтение низкоконтрастного цветного изображения
image = cv2.imread('sample_pict.jpg')

# Получение размеров изображения
rows, cols, channels = image.shape

# Извлечение каналов RGB с помощью функции
red_channel, green_channel, blue_channel = split_channels(image)

# Переписанная функция выравнивания гистограммы
def equalize_histogram(channel):
    histogram, _ = np.histogram(channel.flatten(), bins=256, range=[0, 256])
    cdf = histogram.cumsum() / histogram.sum()
    cdf_min = cdf[np.nonzero(cdf)].min()  # Использование nonzero вместо прямого условия
    equalized_channel = np.interp(channel, np.arange(256), (cdf - cdf_min) / (1 - cdf_min) * 255).astype(np.uint8)
    return equalized_channel

# Применение выравнивания гистограммы к каждому каналу
red_eq = equalize_histogram(red_channel)
green_eq = equalize_histogram(green_channel)
blue_eq = equalize_histogram(blue_channel)

# Изменённая функция Рэлеевского преобразования для упрощения
def apply_rayleigh_transform(channel):
    scale_factor = 255 / np.log1p(255)  # Использование np.log1p для улучшенной стабильности
    transformed_channel = scale_factor * np.log1p(channel.astype(np.float32))
    return np.clip(transformed_channel, 0, 255).astype(np.uint8)

# Применение Рэлеевского преобразования к каждому каналу
red_transformed = apply_rayleigh_transform(red_eq)
green_transformed = apply_rayleigh_transform(green_eq)
blue_transformed = apply_rayleigh_transform(blue_eq)

# Объединение каналов обратно в улучшенное изображение
enhanced_image = cv2.merge([blue_transformed, green_transformed, red_transformed])

# Построение графиков в изменённой компоновке
fig, ax = plt.subplots(3, 4, figsize=(14, 8))

# Исходное изображение
ax[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
ax[0, 0].set_title('Исходное изображение')

# Красный канал и его преобразования
ax[0, 1].imshow(red_channel, cmap='gray')
ax[0, 1].set_title('Красный канал')
ax[0, 2].imshow(red_transformed, cmap='gray')
ax[0, 2].set_title('Красный канал после Рэлеевского преобразования')
ax[0, 3].hist(red_channel.ravel(), bins=256, range=[0, 256], color='r')
ax[0, 3].set_title('Гистограмма красного канала')

# Зелёный канал и его преобразования
ax[1, 1].imshow(green_channel, cmap='gray')
ax[1, 1].set_title('Зелёный канал')
ax[1, 2].imshow(green_transformed, cmap='gray')
ax[1, 2].set_title('Зелёный канал после Рэлеевского преобразования')
ax[1, 3].hist(green_channel.ravel(), bins=256, range=[0, 256], color='g')
ax[1, 3].set_title('Гистограмма зелёного канала')

# Синий канал и его преобразования
ax[2, 1].imshow(blue_channel, cmap='gray')
ax[2, 1].set_title('Синий канал')
ax[2, 2].imshow(blue_transformed, cmap='gray')
ax[2, 2].set_title('Синий канал после Рэлеевского преобразования')
ax[2, 3].hist(blue_channel.ravel(), bins=256, range=[0, 256], color='b')
ax[2, 3].set_title('Гистограмма синего канала')

# Улучшенное изображение
ax[2, 0].imshow(cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB))
ax[2, 0].set_title('Улучшенное изображение')

plt.tight_layout()
plt.show()
