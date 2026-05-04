import asyncio

from fastapi import FastAPI
from contextlib import asynccontextmanager
from aiogram import Bot, Dispatcher, types
from aiogram.filters import CommandStart
from aiogram.enums import ChatAction
import os
from dotenv import load_dotenv
import logging
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from instructions import load_document_text, get_start, get_response, DB_PATH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
bot = Bot(token=os.getenv("BOT_TOKEN"))
dp = Dispatcher()

class AppState:
    def __init__(self):
        self.vector_store = None # Ваш индекс FAISS
        self.llm = None

state = AppState()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Код здесь выполняется ПРИ ЗАПУСКЕ
    ollama_url = os.getenv("OLLAMA_URL")
    embeddings_model = OllamaEmbeddings(model="nomic-embed-text", base_url=ollama_url)
    if os.path.exists(DB_PATH):
        logger.info("Загрузка существующей базы FAISS из файла...")
        # allow_dangerous_deserialization=True нужен для загрузки локальных файлов pickle
        app.state.vector_store = FAISS.load_local(DB_PATH, embeddings_model, allow_dangerous_deserialization=True)
    else:
        doc_id = os.getenv("DOC_ID")
        # Сохраняем данные в state приложения, чтобы не использовать global
        document_text = load_document_text(doc_id)
        logger.info("Документ успешно загружен")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_text(document_text)
        logger.info("Файл базы не найден. Создание нового индекса (это может занять время)...")
        app.state.vector_store = FAISS.from_texts(chunks, embeddings_model)
        # 2. Сохраняем базу в файл
        app.state.vector_store.save_local(DB_PATH)
        logger.info(f"База сохранена в папку {DB_PATH}")
    app.state.llm = OllamaLLM(model="gemma2:2b", base_url=ollama_url)

    async def start_polling():
        await dp.start_polling(bot, handle_signals=False)

    polling_task = asyncio.create_task(start_polling())

    yield

    polling_task.cancel()
    try:
        await polling_task
    except asyncio.CancelledError:
        pass
    await bot.session.close()
    # Код здесь выполняется ПРИ ВЫКЛЮЧЕНИИ
    logger.info("Завершение работы...")

app = FastAPI(lifespan=lifespan)


@dp.message()  # ← ПЕРЕД CommandStart
async def debug_all(message: types.Message):
    logger.info(f"Получено сообщение: '{message.text}' от {message.from_user.id}")
    await message.answer("Debug: работает!")

@dp.message(CommandStart())
async def cmd_start(message: types.Message):
    user_name = message.from_user.first_name
    logger.info(f"User {user_name} triggered /start command")
    await message.answer(get_start(user_name), parse_mode="HTML")

@dp.message()
async def echo_all(message: types.Message):
  await bot.send_chat_action(message.chat.id, ChatAction.TYPING)
  await message.answer(get_response(app.state.llm, app.state.vector_store, message.text))

@app.get("/")
async def root():
    return {"status": "Bot started..."}
