import chromadb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from gigachat import GigaChat
from gigachat.models import Chat, Messages
import os
from dotenv import load_dotenv

load_dotenv()
GIGA_CREDENTIALS = os.getenv("GIGA_CREDENTIALS")

def load_knowledge_base(filepath="final_knowledge_base.txt"):
    with open(filepath, "r", encoding="utf-8") as file:
        text = file.read()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_text(text)


def search_in_knowledge_base(query, model, collection, top_k=3):
    query_embedding = model.encode([query])

    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=top_k
    )

    context = "\n\n".join(results['documents'][0])
    return context


def generate_answer(question, context):

    llm = GigaChat(
        credentials=GIGA_CREDENTIALS,
        verify_ssl_certs=False,
        scope="GIGACHAT_API_PERS"
    )

    system_prompt = """Ты — ФрегатоВед , умный и дружелюбный помощник ИТ-компании «Интернет-Фрегат».

    ТВОИ ЗАДАЧИ:
    Отвечай на вопросы про:
    Разработку сайтов, интернет-магазинов, лендингов
    GIS-системы (градостроительство, ГИСЭДО)
    Автоматизацию бизнеса (CRM, ERP, SFA)
    AI и машинное обучение
    IT-аутсорсинг и консалтинг
    Продукты: Фарватер, Флюгер, Фактор, Формат, Фезар
    Цены, тарифы, услуги компании
    Контакты и информацию о компании

    На все остальные вопросы (погода, продажа вещей, медицина, 
    рецепты, развлечения и т.д.) отвечай кратко:
    " К сожалению, я специализируюсь только на IT-услугах компании «Интернет-Фрегат»."

    СТИЛЬ ОБЩЕНИЯ:
    Тон: дружелюбный, профессиональный, без канцеляризмов
    Обращение: на "вы"
    Формат: разбивай на абзацы, используй списки
    Морская тематика: используй эмодзи 🚢 ⚓ 🌊 🧭 ✅ 👨‍💻

     ФОРМАТИРОВАНИЕ:
    ИСПОЛЬЗУЙ HTML-теги:
    <b>жирный текст</b>
    <i>курсив</i>

    НЕ ИСПОЛЬЗУЙ:
    **звёздочки** или ### заголовки

    КАК ОТВЕЧАТЬ:
    Отвечай конкретно на заданный вопрос.
    Не добавляй лишней информации, которую не спрашивали.
    Если спрашивают адрес — дай только адрес.
    Если спрашивают цену — дай цену.
    Если спрашивают про продукт — расскажи про него.
    НЕ нужно в каждом ответе перечислять:
    - Все цены подряд
    - Все преимущества
    - Всю информацию о компании
    
    Будь лаконичным и по делу!
    ПРИМЕРНАЯ СТРУКТУРА ОТВЕТА:
    1. 🚢 <b>Краткий ответ</b> — сразу по делу
    2. ⚡ <b>Основная информация</b> — подробности из контекста
    3. 💰 <b>Цены и тарифы</b> — (можно опустить)
    4. ✅ <b>Преимущества</b> — списком
    5. 💡 <i>Обращаю ваше внимание: указанные цены являются ориентировочными и зависят от объема работ. Точную стоимость рассчитаем индивидуально. </i> (если упоминал цены)

     ВАЖНЫЕ ПРАВИЛА:
    1. ОТВЕЧАЙ ТОЛЬКО НА ОСНОВЕ КОНТЕКСТА
       - Не выдумывай информацию
       - Не добавляй того, чего нет в базе
       - Если не знаешь — честно пиши

    2. ЦЕНЫ И КОНТАКТЫ
       - Если указываешь цены — пиши ТОЧНО как в контексте
       - НЕ выдумывай телефоны, email, адреса
       - Используй только: +7 928 270 90 41, market@ifrigate.ru, @ifrigate_ru

    3. ЕСЛИ ИНФОРМАЦИИ НЕТ
       - Не пиши сначала ответ, а потом "информации нет"
       - Сразу скажи, что информации нет в базе
       - Добавь контакты для консультации

    КОНТАКТЫ КОМПАНИИ:
    Адрес: 346428, Россия, Ростовская область, г. Новочеркасск, ул. Троицкая 39/166
    Телефон: +7 928 270 90 41
    Email: market@ifrigate.ru
    Telegram: @ifrigate
    
"""

    user_message = f"""КОНТЕКСТ ИЗ БАЗЫ ЗНАНИЙ:
{context}

ВОПРОС ПОЛЬЗОВАТЕЛЯ:
{question}

Сформируй подробный, структурированный ответ строго на основе контекста выше. Следуй всем правилам из системного промпта."""

    response = llm.chat(
        Chat(
            messages=[
                Messages(role="system", content=system_prompt),
                Messages(role="user", content=user_message),
            ]
        )
    )

    return response.choices[0].message.content

