# ФрегатоВед — Telegram-бот для компании «Интернет-Фрегат»

Интеллектуальный помощник на базе RAG-архитектуры, который отвечает на вопросы клиентов об услугах, продуктах и ценах компании.

## Технологии

- Python 3.14
- aiogram 3.x (Telegram API)
- SentenceTransformers (rubert-tiny2)
- ChromaDB (векторное хранилище)
- GigaChat API (генерация ответов)
- LangChain (разбиение текста)

## Структура проекта

- `bot_brain.py` — RAG-логика (поиск + генерация)
- `telegram_bot.py` — интерфейс Telegram-бота
- `chunking.py` — разбиение текста на чанки
- `vectorize.py` — векторизация и сохранение в ChromaDB
- `compare_metrics.py` — сравнительный анализ метрик
- `final_knowledge_base.txt` — база знаний

## Установка

1. Клонируйте репозиторий
2. Создайте файл `.env` с токенами:
   BOT_TOKEN=ваш_токен_бота
   GIGA_CREDENTIALS=ваш_ключ_gigachat
3. Установите зависимости:
   pip install aiogram sentence-transformers chromadb langchain gigachat python-dotenv
4. Запустите бота
   

Проект в рамках производственной практики
