import telebot
import model_handler
import os
from flask import Flask, request
import logging

TOKEN = os.environ.get('TELEGRAM_TOKEN')
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

if "HEROKU_APP_NAME" in list(os.environ.keys()):
	logger = telebot.logger
	telebot.logger.setLevel(logging.INFO)

	server = Flask(__name__)
	@server.route("/bot", methods=['POST'])
	def getMessage():
		bot.process_new_updates([telebot.types.Update.de_json(request.stream.read().decode("utf-8"))])
		return "!", 200
	@server.route("/")
	def webhook():
		bot.remove_webhook()
		bot.set_webhook(url="https://"+os.environ.get("HEROKU_APP_NAME")+".herokuapp.com/bot")
		return "?", 200
	server.run(host="0.0.0.0", port=os.environ.get('PORT', 80))
else:
	bot.remove_webhook()
	bot.polling(none_stop=True)
