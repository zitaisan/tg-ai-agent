import asyncio
import os
import httpx
from aiogram import Bot, Dispatcher, F
from aiogram.enums import ChatAction
from aiogram.filters import CommandStart
from aiogram.types import Message, ReplyKeyboardMarkup, KeyboardButton
from dotenv import load_dotenv
MAX_MESSAGE_LENGTH = 4000


async def send_long_message(message: Message, text: str):
    """
    Telegram принимает максимум ~4096 символов.
    """
    if not text:
        text = "Пустой ответ."

    for i in range(0, len(text), MAX_MESSAGE_LENGTH):
        await message.answer(text[i:i + MAX_MESSAGE_LENGTH])


async def ask_rag(text: str) -> str:

    async with httpx.AsyncClient(timeout=120) as client:

        response = await client.post(
            API_URL,
            json={
                "text": text,
                "mode": "agent_pattern"
            }
        )

    print("=" * 80)
    print("STATUS:", response.status_code)
    print(response.text[:1000])
    print("=" * 80)

    response.raise_for_status()

    data = response.json()

    answer = (
        data.get("answer")
        or data.get("response")
        or data.get("text")
    )

    if answer:
        return answer

    # если API внезапно поменяется
    import json

    return json.dumps(
        data,
        ensure_ascii=False,
        indent=2
    )


load_dotenv()

BOT_TOKEN = os.getenv("TELEGRAM_TOKEN")
API_URL = os.getenv("API_URL")


bot = Bot(BOT_TOKEN)
dp = Dispatcher()


# ---------- UI кнопки ----------

main_keyboard = ReplyKeyboardMarkup(
    keyboard=[
        [
            KeyboardButton(text="💬 Написать запрос")
        ],
        [
            KeyboardButton(text="🔄 Новый запрос"),
            KeyboardButton(text="ℹ️ Помощь")
        ]
    ],
    resize_keyboard=True
)


# ---------- запрос к RAG API ----------

async def ask_rag(text: str):

    async with httpx.AsyncClient(timeout=120) as client:

        response = await client.post(
            API_URL,
            json={
                "text": text,
                "mode": "agent_pattern"
            }
        )

        response.raise_for_status()

        data = response.json()

        return (
            data.get("answer")
            or data.get("response")
            or data.get("text")
            or str(data)
        )


# ---------- /start ----------

@dp.message(CommandStart())
async def start(message: Message):

    await message.answer(
        f"""
👋 Привет, {message.from_user.first_name}!

Я AI-ассистент на базе Agentic Graph RAG.

Я могу:
• отвечать на вопросы по базе знаний;
• искать информацию;
• анализировать документы.

Выберите действие:
        """,
        reply_markup=main_keyboard
    )


# ---------- кнопка запроса ----------

@dp.message(F.text == "💬 Написать запрос")
async def request_button(message: Message):

    await message.answer(
        "✍️ Напишите ваш вопрос:",
        reply_markup=main_keyboard
    )


# ---------- помощь ----------

@dp.message(F.text == "ℹ️ Помощь")
async def help_button(message: Message):

    await message.answer(
        """
ℹ️ Как пользоваться:

1. Нажмите "💬 Написать запрос"
2. Отправьте вопрос
3. Получите ответ AI

Пример:
"Какие документы есть в базе?"
        """,
        reply_markup=main_keyboard
    )


# ---------- новый запрос ----------

@dp.message(F.text == "🔄 Новый запрос")
async def new_request(message: Message):

    await message.answer(
        "Введите новый вопрос 👇",
        reply_markup=main_keyboard
    )


# ---------- обработка вопросов ----------

@dp.message(F.text)
async def handle_message(message: Message):

    if message.text.startswith(("💬", "ℹ️", "🔄")):
        return

    # показывает "печатает..."
    await bot.send_chat_action(
        chat_id=message.chat.id,
        action=ChatAction.TYPING
    )

    wait_message = await message.answer("⏳ Ищу информацию...")

    try:

        answer = await ask_rag(message.text)

        print(f"Ответ длиной {len(answer)} символов")

        await wait_message.delete()

        await send_long_message(message, answer)

    except httpx.HTTPStatusError as e:

        await wait_message.delete()

        await message.answer(
            f"❌ Ошибка API\n"
            f"HTTP {e.response.status_code}\n\n"
            f"{e.response.text[:1000]}"
        )

    except httpx.TimeoutException:

        await wait_message.delete()

        await message.answer(
            "⏳ Сервис слишком долго отвечает."
        )

    except Exception as e:

        await wait_message.delete()

        await message.answer(
            f"❌ Ошибка:\n{e}"
        )


async def main():

    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())