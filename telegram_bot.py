import asyncio
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command
from aiogram.types import Message, ReplyKeyboardMarkup, KeyboardButton, InlineKeyboardMarkup, InlineKeyboardButton

from bot_brain import load_knowledge_base, search_in_knowledge_base, generate_answer
from sentence_transformers import SentenceTransformer
import chromadb
import re

import os
from dotenv import load_dotenv

load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN")

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

model = None
collection = None

main_menu = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton(text="🛠 Услуги"), KeyboardButton(text="📦 Продукты")],
        [KeyboardButton(text="💰 Прайс-лист"), KeyboardButton(text="📞 Контакты")],
        [KeyboardButton(text="❓ Помощь")]
    ],
    resize_keyboard=True,
    input_field_placeholder="Выберите раздел или задайте вопрос..."
)

feedback_keyboard = InlineKeyboardMarkup(inline_keyboard=[
    [
        InlineKeyboardButton(text="👍 Ответ полезен", callback_data="feedback_good"),
        InlineKeyboardButton(text="👎 Ответ не помог", callback_data="feedback_bad")
    ],
    [
        InlineKeyboardButton(text="📞 Заказать звонок", callback_data="callback_request")
    ]
])

@dp.message(Command("start"))
async def cmd_start(message: Message):
    welcome_text = """
🚢 <b>Добро пожаловать на борт!</b>

Я — <b>ФрегатоВед</b>, ваш умный навигатор по базе знаний компании «Интернет-Фрегат». 

Я здесь, чтобы быстро и точно отвечать на вопросы о наших продуктах, услугах и технологиях.

⚙️ <b>Как мы будем работать:</b>
1. Задавайте любой вопрос о продуктах или услугах компании.
2. Не бойтесь писать своими словами, использовать профессиональный сленг или допускать опечатки — мой семантический поиск всё поймёт.
3. Я ищу ответы *только* в официальной документации. Если ответа нет в базе, я честно признаюсь, чтобы не вводить вас в заблуждение.

<b>Жду ваш первый вопрос! 👇</b>
"""
    await message.answer(welcome_text, parse_mode="HTML", reply_markup=main_menu)

@dp.message(Command("menu"))
async def cmd_menu(message: Message):
    await message.answer(" Выберите интересующий раздел:", reply_markup=main_menu)


@dp.message(F.text == "🛠 Услуги")
async def show_services(message: Message):
    services_text = """
🚢 <b>Услуги компании «Интернет-Фрегат»</b>

Наша компания предлагает широкий спектр услуг для автоматизации вашего бизнеса и создания эффективных цифровых решений.

⚡ <b>Основные направления:</b>

<b>1. 📱 Разработка сайтов</b>
• Корпоративные сайты (Интернет-ЯХТА, БРИГАНТИНА, ФАРВАТЕР)
• Интернет-магазины (Старт, Малый бизнес, Маркетплейс)
• Создание лендингов (от 25 000 ₽)
• Лендинги на Tilda (от 30 000 ₽)
• Государственные порталы

<b>2. 🗺️ Разработка GIS</b>
• Градостроительные ГИС (от 700 000 ₽)
• ГИСЭДО (геоинформационные системы электронного документооборота)
• ГИС для управления территориями и активами

<b>3. 🤖 Автоматизация бизнеса</b>
• Автоматизация продаж и мерчандайзинга (SFA)
• CRM/ERP системы
• Логистика и складской учет (WMS, TMS)
• BI-аналитика

<b>4. 💻 IT-аутсорсинг и консалтинг</b>
• IT-аудит
• IT-консалтинг (3 500 ₽/час)
• QA и тестирование
• Разработка MVP (от 2-4 месяцев)

<b>5. 🔧 Сопровождение и поддержка</b>
• Сопровождение сайтов (от 3 000 ₽/мес)
• Регистрация и сопровождение доменов (от 500 ₽)
• Размещение на хостинге (от 11 400 ₽/год)
• Продвижение сайтов (SEO)
• Контекстная реклама
• Перенос публикаций из Telegram в MAX (от 1 000 ₽)

<b>6. 🎨 Дизайн и графика</b>
• Веб-дизайн
• UX/UI дизайн
• Фирменный стиль

<b>7. 🧠 Искусственный интеллект (AI)</b>
• AI-консалтинг
• Разработка моделей машинного обучения
• Интеграция AI в существующие системы
• Поддержка и сопровождение ИИ-систем

---
💡 <i>Указанные цены являются ориентировочными. Точную стоимость рассчитаем индивидуально после уточнения деталей вашего проекта.</i>

---
👨‍💻⚓ <b>Для точного расчёта и консультации:</b>
📞 +7 928 270 90 41
📧 market@ifrigate.ru
💬 @ifrigate

🚢🌊 Напишите «<b>Хочу консультацию</b>» — перезвоним!
"""
    await message.answer(services_text, parse_mode="HTML", reply_markup=feedback_keyboard)


@dp.message(F.text == "📦 Продукты")
async def show_products(message: Message):
    products_text = """
🚢⚓ <b>Готовые программные продукты «Интернет-Фрегат»</b>

Мы создаем готовые ИТ-решения, способные решать самые сложные бизнес-задачи.

⚡ <b>ЛИНЕЙКА «ФАРВАТЕР» (GIS-системы):</b>

<b>💰 Фарватер-Активы</b> — от 5 000 000 ₽
Геоинформационная система мониторинга, анализа и контроля имущества и территории организации

<b>💰 Фарватер-ГИСОГД</b> — 7 000 000 ₽
Готовое программное решение с функциями автоматизации предоставления государственных и муниципальных услуг

<b>📋 Фарватер-ГИСдок</b>
Геоинформационная система электронного документооборота (ГИСЭДО)

<b>📋 Фарватер-ИСОГД</b>
Автоматизированная информационная система обеспечения градостроительной деятельности (АИСОГД)

<b>📋 Фарватер-ГИСОД</b>
Геоинформационная система обеспечения деятельности организации

⚡ <b>ЛИНЕЙКА «ФЛЮГЕР» (Автоматизация продаж):</b>

<b>💰 Флюгер-Продажи</b> — от 5 250 ₽ за 1 мобильное рабочее место
Система автоматизации продаж и мерчандайзинга с мобильными торговыми командами (SFA)

<b>📋 Флюгер-Логистика</b>
Автоматизация транспортного логистического центра (TMS)

<b>📋 Флюгер-Склад</b>
Система автоматизации складского хозяйства (WMS)

⚡ <b>ЛИНЕЙКА «ФАКТОР» (Аналитика):</b>

<b>💰 Платформа «Фактор»</b> — 2 150 000 ₽
Сбор и анализ информации, реестры, расчет показателей и гео-аналитика

⚡ <b>ЛИНЕЙКА «ФОРМАТ» (Госуслуги):</b>

<b>📋 Формат</b>
Система автоматизации предоставления услуг (САПУ) и регламентов межведомственного взаимодействия (МЭВ)

⚡ <b>ЛИНЕЙКА «ФЕЗАР» (CMS):</b>

<b>📋 Фезар-CMS / госCMS</b>
Корпоративный сайт для успешного бизнеса; официальный сайт государственного ведомства

---
💡 <i>Указанные цены являются ориентировочными и зависят от объема работ, дополнительных модулей и требований к интеграции. Точную стоимость рассчитаем индивидуально.</i>

---
👨‍⚓ <b>Для точного расчёта и консультации:</b>
📞 +7 928 270 90 41
📧 market@ifrigate.ru
💬 @ifrigate

🚢🌊 Напишите «<b>Хочу консультацию</b>» — перезвоним!
"""
    await message.answer(products_text, parse_mode="HTML", reply_markup=feedback_keyboard)


@dp.message(F.text == "💰 Прайс-лист")
async def show_pricelist(message: Message):
    pricelist_text = """
🚢 <b>ПРАЙС-ЛИСТ НА УСЛУГИ И ПРОДУКТЫ</b>

💰 <b>РАЗРАБОТКА САЙТОВ:</b>

<b>Корпоративные сайты:</b>
• Интернет-ЯХТА — от 49 900 ₽
• Интернет-БРИГАНТИНА — от 89 900 ₽
• Интернет-ФАРВАТЕР — от 149 900 ₽

<b>Интернет-магазины:</b>
• Старт — от 300 000 ₽
• Малый бизнес — от 500 000 ₽
• Маркетплейс — от 2 500 000 ₽

<b>Лендинги:</b>
• Базовый вариант — от 25 000 ₽
• Лендинги на Tilda — от 30 000 ₽

💰 <b>РАЗРАБОТКА GIS:</b>
• Базовый вариант — от 700 000 ₽

💰 <b>IT-АУТСОРСИНГ:</b>
• Основной рейт — 3 500 ₽ / час

💰 <b>СОПРОВОЖДЕНИЕ И ПОДДЕРЖКА:</b>

<b>Сопровождение сайтов:</b>
• Тариф Старт — от 3 000 ₽ / мес

<b>Регистрация доменов:</b>
• Тариф Базовый — от 500 ₽

<b>Размещение на хостинге:</b>
• Тариф ХП 1K (до 1 ГБ) — 11 400 ₽ / год
• Тариф ХП 3K (до 3 ГБ) — 14 160 ₽ / год
• Тариф ХП 5K (до 5 ГБ) — 21 360 ₽ / год
• Тариф ХП 10K (до 10 ГБ) — 27 600 ₽ / год
• Тариф ХП 100K (до 100 ГБ) — 63 600 ₽ / год

<b>Перенос Telegram в MAX:</b>
• Тариф ТелегМАХ ТМ-1 (до 1500 сообщений) — от 1 000 ₽
• Тариф ТелегМАХ ТМ-2 (до 2500 сообщений) — от 2 500 ₽
• Тариф ТелегМАХ ТМ-3 (более 3000 сообщений) — от 3 000 ₽

💰 <b>ГОТОВЫЕ ПРОДУКТЫ:</b>
• Фарватер-Активы — от 5 000 000 ₽
• Фарватер-ГИСОГД — 7 000 000 ₽
• Платформа «Фактор» — 2 150 000 ₽
• Флюгер-Продажи — от 5 250 ₽ за 1 рабочее место

---
💡 <b>ВАЖНОЕ ПРИМЕЧАНИЕ:</b>
<i>Указанные цены являются ориентировочными и зависят от:
• Объема работ
• Дополнительных модулей
• Требований к интеграции
• Количества пользовательских лицензий

Точную стоимость мы рассчитаем индивидуально для вашего проекта после консультации.</i>

---
👨‍💻⚓ <b>Для точного расчёта и консультации:</b>
📞 +7 928 270 90 41
📧 market@ifrigate.ru
💬 @ifrigate

🚢 Напишите «<b>Хочу консультацию</b>» — перезвоним!
"""
    await message.answer(pricelist_text, parse_mode="HTML", reply_markup=feedback_keyboard)

@dp.message(F.text == "📞 Контакты")
async def show_contacts(message: Message):
    contacts_text = """
🚢 <b>Контакты компании «Интернет-Фрегат»</b>

⚓ Мы на связи 24/7 и готовы ответить на любые ваши вопросы!

🌊 <b>Свяжитесь с нами:</b>
📞 Телефон: <b>+7 928 270 90 41</b>
📧 Email: <b>market@ifrigate.ru</b>
💬 Telegram: @ifrigate

📱 <b>Мы в соцсетях:</b>
• Telegram: @ifrigate_ru
• Telegram: @itFreeGate
• VK: @ifrigate

---
🏢 <b>О компании:</b>
✅ Работаем с <b>2000 года</b> (26 лет на рынке)
✅ Реализовано <b>500+ проектов</b>
✅ Аккредитованная ИТ-компания <b>Минцифры РФ</b> (с 2010 года)
✅ Член Ассоциации <b>РУССОФТ</b> (с 2003 года)
---
👨‍💻 <b>Задайте любой вопрос — мы поможем сориентироваться в услугах, продуктах и ценах!</b>

🚢🌊 Или просто напишите «<b>Хочу консультацию</b>» — наш менеджер перезвонит!
"""
    await message.answer(contacts_text, parse_mode="HTML", reply_markup=feedback_keyboard)


@dp.message(F.text == "❓ Помощь")
async def show_help(message: Message):
    help_text = """
🚢 <b>Помощь</b>

Просто напишите мне любой вопрос о компании, услугах или ценах. Например:
• <i>Сколько стоит сайт?</i>
• <i>Что такое Флюгер?</i>
• <i>Какие AI-услуги вы предоставляете?</i>
• <i>Как заказать звонок?</i>

Или выберите раздел в меню ниже 👇
"""
    await message.answer(help_text, parse_mode="HTML", reply_markup=main_menu)

@dp.message()
async def handle_message(message: Message):
    user_question = message.text.lower()

    irrelevant_keywords = [
        'продать телефон', 'продажа телефона', 'как продать',
        'купить телефон', 'где купить телефон',
        'скорую', 'скорая помощь', 'вызвать скорую',
        'полицию', 'пожарных', '112', '103',
        'больницу', 'врача', 'медицинскую помощь',
        'рецепт', 'приготовить', 'как приготовить',
        'погода', 'какая погода',
        'фильм', 'кино', 'посмотреть фильм',
        'музыка', 'песня',
        'новости', 'политика', 'президент',
        'спорт', 'футбол', 'хоккей',
        'игра', 'играть',
        'знакомства', 'любовь',
        'здоровье', 'болезнь',
        'ремонт квартиры', 'как сделать ремонт',
        'как заработать', 'работа', 'вакансия',
        'обучение', 'курсы', 'университет',
        'путешествие', 'отпуск', 'отель',
        'еда', 'ресторан', 'кафе',
        'машина', 'автомобиль', 'купить машину',
        'недвижимость', 'квартира', 'дом', 'аренда',
        'животные', 'кот', 'собака',
        'мода', 'одежда', 'обувь',
        'косметика', 'красота',
    ]

    if any(keyword in user_question for keyword in irrelevant_keywords):
        refusal_text = "🚢 К сожалению, я специализируюсь только на IT-услугах компании «Интернет-Фрегат»."
        await message.answer(refusal_text, reply_markup=feedback_keyboard)
        return

    status_msg = await message.answer("🔍 Ищу ответ в базе знаний...")

    try:
        context = search_in_knowledge_base(user_question, model, collection, top_k=8)

        if not context or len(context) < 50:
            await status_msg.delete()
            await message.answer(
                "🚢 <b>К сожалению, в базе знаний компании нет информации по вашему вопросу.</b>\n\n"
                "🌊 <i>Однако наши специалисты с удовольствием ответят на него лично!</i>\n\n"
                "---\n"
                "👨 <b>Контакты:</b>\n"
                "📞 +7 928 270 90 41\n"
                "📧 market@ifrigate.ru\n"
                "💬 @ifrigate\n\n"
                "🚢 Напишите «<b>Хочу консультацию</b>» — перезвоним!",
                parse_mode="HTML",
                reply_markup=feedback_keyboard
            )
            return

        answer = generate_answer(user_question, context)

        has_contacts = (
                '+7 928 270 90 41' in answer or
                'market@ifrigate.ru' in answer or
                '@ifrigate' in answer or
                'консультации' in answer.lower()
        )

        if not has_contacts:
            contacts_block = """

---
‍💻 <b>Для точного расчёта и консультации:</b>
📞 +7 928 270 90 41
📧 market@ifrigate.ru
💬 @ifrigate

🚢🌊 Напишите «<b>Хочу консультацию</b>» — перезвоним!"""
            answer += contacts_block

        answer = re.sub(r'\n{3,}', '\n\n', answer)
        answer = '\n'.join(line.strip() for line in answer.split('\n'))

        await status_msg.delete()

        try:
            await message.answer(answer, parse_mode="HTML", reply_markup=feedback_keyboard)
        except Exception:
            clean_answer = re.sub(r'<[^>]+>', '', answer)
            await message.answer(clean_answer, reply_markup=feedback_keyboard)

    except Exception as e:
        try:
            await status_msg.delete()
        except Exception:
            pass
        await message.answer("⚠️ Произошла ошибка при обработке вашего вопроса. Попробуйте ещё раз.")
        print(f"Ошибка: {e}")


@dp.callback_query(F.data.in_(["feedback_good", "feedback_bad", "callback_request"]))
async def process_feedback(callback_query: types.CallbackQuery):

    if callback_query.data == "feedback_good":
        await callback_query.answer("Спасибо! Рады, что помогли вам! 🚢⚓", show_alert=True)

    elif callback_query.data == "feedback_bad":
        await callback_query.answer("Спасибо за отзыв. Мы работаем над улучшением! 🌊", show_alert=True)

    elif callback_query.data == "callback_request":
        await callback_query.answer("Заявка принята! Менеджер скоро свяжется с вами. 📞", show_alert=True)
        await callback_query.message.answer(
            "👨‍💻⚓ <b>Отлично!</b> Наш менеджер уже получил сигнал.\n\n"
            "Вы также можете написать нам напрямую:\n"
            "📞 +7 928 270 90 41\n"
            "📧 market@ifrigate.ru\n"
            "💬 @ifrigate",
            parse_mode="HTML"
        )
    await callback_query.message.edit_reply_markup(reply_markup=None)

async def main():
    global model, collection
    print("Запуск Telegram-бота...")
    model = SentenceTransformer('cointegrated/rubert-tiny2')
    chunks = load_knowledge_base("final_knowledge_base.txt")
    client = chromadb.Client()
    collection = client.create_collection(name="ifrigate_knowledge")
    embeddings = model.encode(chunks)
    collection.add(
        embeddings=embeddings.tolist(),
        documents=chunks,
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )
    print(f"База знаний готова. Загружено {len(chunks)} чанков.\n")
    print("Бот запущен!")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())