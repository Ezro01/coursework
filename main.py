import logging
from telegram import Update, ReplyKeyboardMarkup
from telegram.ext import (
    Updater,
    CommandHandler,
    MessageHandler,
    filters,
    CallbackContext,
    ApplicationBuilder
)

from model_predict import load_and_preprocess_data, predict_sales

# Настройка логов
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

async def start(update: Update, context: CallbackContext) -> None:
    """Приветственное сообщение с кнопкой"""
    user = update.effective_user
    welcome_text = (
        f"Привет, {user.first_name}! 👋\n"
        "Я бот для прогнозирования продаж.\n"
        "Нажми кнопку ниже, чтобы получить прогноз на слудующие 7 дней!"
    )
    reply_keyboard = [["Сделать прогноз"]]
    await update.message.reply_text(
        welcome_text,
        reply_markup=ReplyKeyboardMarkup(reply_keyboard, resize_keyboard=True)
    )

async def make_prediction(update: Update, context: CallbackContext) -> None:
    """Обработка нажатия кнопки"""
    await update.message.reply_text("⏳ Загружаю данные и делаю прогноз...")

    try:
        # Загрузка данных и предсказание
        df, next_date = load_and_preprocess_data()
        df_predict = df[df['Дата'] == next_date]
        result = predict_sales(df_predict)

        # Форматируем результат для вывода
        output = "📊 Прогноз продаж на 7 дней вперёд:\n"
        for _, row in result.iterrows():
            output += f"Товар: {row['Товар']} | Прогноз: {row['Прогноз']} шт.\n"

        await update.message.reply_text(output)
    except Exception as e:
        logger.error(f"Ошибка: {e}")
        await update.message.reply_text("❌ Произошла ошибка. Попробуйте позже.")

def main():
    """Запуск бота"""
    # Новый способ инициализации (для PTB v20.x+)
    application = ApplicationBuilder().token("7450398355:AAFWD1hOF9kDBdhhkrmyhlLl9uD5-2SASI4").build()

    # Обработчики команд
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & filters.Regex("^Сделать прогноз$"), make_prediction))

    # Запуск бота
    application.run_polling()

if __name__ == '__main__':
    main()