### Решение дипломного задания по нейронным сетям Skillbox

Ссылка на google colab: https://colab.research.google.com/drive/1VjMjXh08s8_cau1rUcmNe1V9bqESaHHn?usp=sharing

Точность на тестсете: 0.37

Время инференса на ноутбуке с видеокартой GT 1050 составляет менее 0.1 с.

До запуска веб-камеры сначала нужно скачать модель с моего Google Drive путем запуска файла load_model.py.

Программа рассчитана на распознавание одного лица в кадре.

Зависимости load_model.py
gdown

Зависимости webcam.py:
numpy
tensorflow
keras
opencv-python
