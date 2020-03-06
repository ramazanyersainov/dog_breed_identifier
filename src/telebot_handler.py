import telebot
import model_handler
import os

TOKEN = 'YOUR TOKEN'
bot = telebot.TeleBot(TOKEN)
model = model_handler.get_model()

@bot.message_handler(commands=['start'])
def start_message(message):
    bot.send_message(message.chat.id, 'Hi! Send me a doggie photo and I\'ll try to name its\' breed')

@bot.message_handler(content_types=['photo'])
def handle_docs_photo(message):
	try:
		file_info = bot.get_file(message.photo[len(message.photo)-1].file_id)
		downloaded_file = bot.download_file(file_info.file_path)
		src='tmp/'+file_info.file_path;
		with open(src, 'wb') as new_file:
			new_file.write(downloaded_file)
		result = model_handler.predict_image(model, src)
		os.remove(src) if os.path.exists(src) else None
		bot.reply_to(message, "Looks like {}".format(result.replace('_',' ')))
	except:
		bot.reply_to(message, "Something went wrong")


@bot.message_handler(content_types=['text'])
def send_text(message):
    bot.reply_to(message, "To use the bot, send a doggie photo")


@bot.message_handler(content_types=['document'])
def send_text(message):
    bot.reply_to(message, "Please send as a photo, not as a document")

bot.polling()