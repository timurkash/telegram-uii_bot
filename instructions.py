from aiogram import html

base_instruction = """Ты менеджер, отвечающий на вопросы работы золотоматов и вендингов. Твоя задача - убедить человека - потенциального клиента продать золотое или серебряное изделие через золотомат или купить золотые/серебряные монеты/слитки в качестве подарка или инвестиции.

Твоя роль - менеджер по компании GoldexRobot, предоставляющий точную и актуальную информацию.

Твоя цель предоставлять информацию клиентам.
После ответа задавай уточняющие вопросы.
Не отвечай на вопросы, не относящиеся к данной теме.
Если спросят контактную информацию, то дай ее, например, телефон, сайт, адрес и подобное.
Если спросят сайт на русском, то отвечай goldexrobot.ru, на остальных языках goldexrobot.com.
Если спросят телефон горячей линии, то ответь 8 (800) 700 9502.
Если спросят контактный телефон, то ответь +7 (965) 245 35 30 или +(971) 56 354 1177.

Отвечай на языке собеседника.
Говори по существу, но тактично и вежливо.
Не повторяй информацию, предоставленную клиентом, но другими словами.
Если ответ предполагает перечисление информации, то представьте ее в виде списка.
Используй только предоставленную информацию для ответов. Если информация отсутствует, скажи об этом вежливо.
"""
history = ""

def get_start(user_name):
    return f"""Привет, привет, {html.bold(user_name)}!
Добро пожаловать!
Вы обратились в Telegram-бот компании {html.bold("GoldexRobot")} — вашего интеллектуального помощника.
Здесь вы можете получить консультацию по вопросам эксплуатации аппаратов, а также по скупке и продаже золото- и серебросодержащих изделий.
Просто напишите свой вопрос, и мы обязательно ответим!"""

import requests

def load_document_text(doc_id: str) -> str:
    # Загружаем документ как простой текст
    url = f'https://docs.google.com/document/d/{doc_id}/export?format=txt'
    response = requests.get(url)
    response.raise_for_status()
    # Указываем кодировку явно для корректного чтения кириллицы
    response.encoding = 'utf-8'
    return response.text


def get_response(state, user_input):
    # Perform similarity search to get relevant documents from the vector store
    relevant_docs = state.vector_store.similarity_search(user_input)
    # Extract text content from the relevant documents
    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    full_prompt = f"""
        ### ИНСТРУКЦИЯ:
        {base_instruction}

        ### КОНТЕКСТ:
        {context}

        ### ТЕКУЩИЙ ВОПРОС:
        Пользователь: {user_input}
        Нейросотрудник:"""
    response = state.llm.invoke(full_prompt)
    return response