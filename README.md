<h1> Тестовое задание VK </h1>


В данном репозитории расположено решение вступительной задачи для стажировки в VK
<p align="left">
  <a href="https://github.com/svyatoslav-rozhdestvenskiy">
    <img alt="Static Badge" src="https://img.shields.io/badge/vk_ml_spam_filter-%23000000?style=plastic&label=svyatoslav-rozhdestvenskiy&labelColor=%23008000">
  </a>
  <img alt="GitHub top language" src="https://img.shields.io/github/languages/top/svyatoslav-rozhdestvenskiy/vk_ml_spam_filter?style=plastic&logoColor=008000&labelColor=008000&color=000000">
  <img alt="GitHub repo size" src="https://img.shields.io/github/repo-size/svyatoslav-rozhdestvenskiy/vk_ml_spam_filter?style=plastic&labelColor=008000&color=000000">



# Постановка задачи

Дан тренировочный датасет с текстами сообщений из мессенджера на английском языке. Для каждого из них проставлен флаг того, является ли сообщение СПАМом.

Так же дан тестовый датасет с такими же текстами сообщений, но без флага. На нем нужно будет проскорить модель и приложить результаты. Данные лежат по [ссылке](https://drive.google.com/drive/folders/1f7wUd0gcJpVdFSrZAVwFS8Fxxnkzgu4D?usp=sharing)

Поля датасета:
- text_type - целевая переменная, флаг СИАМ/не СИАМ
- text - текст сообщения.

Задача:
- провести базовую аналитику по имеющимся данным,
- обучить модель по тексту сообщения определить, является ли ее содержимое СПАМом (ожидается, что будут опробованы несколько подходов, из которых аргументированно выбирается наилучший; можно использовать любую библиотеку или фреймворк),
- целевой метрикой при оценке работы модели будет ROC-AUC score,
- провести скоринг лучшей модели по тестовых данных, а результат записать в csv-файл в виде таблицы с колонками score и text;
- выложить код на jupyter notebook и результирующий файл со скорами модели на https://github.com отдельным проектом и поделиться ссылкой в поле для ответа.

# Решение

Решение представлено в ноутбуке spam_filter.ipynb

Также имеется файл solution.py в котором представлен код из ноутбука