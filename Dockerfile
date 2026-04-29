# Используем стабильную версию Python
FROM python:3.11-slim

# Устанавливаем рабочую директорию
WORKDIR /app

# Устанавливаем системные зависимости для работы библиотек (FAISS, доп. пакеты)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Копируем файл зависимостей
COPY requirements.txt .

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем все файлы проекта
COPY . .

# Создаем папку для индекса FAISS, если ее еще нет (чтобы Docker не ругался на права)
RUN mkdir -p faiss_index

# Выставляем порт для FastAPI
EXPOSE 8000

# Команда для запуска (используем uvicorn для FastAPI)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
