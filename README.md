# Dog Breed Identification Bot

Image classification using transfer learning. Based on [Udacity's dog dataset and and Xception model bottleneck features](https://github.com/udacity/dog-project). The first layer of model takes these features as an input, applies a regularization and droupout layers and outputs 133 different classes (dog breeds). Test set accuracy results in 79%.

Before passing data to the model, it uses ResNet50 model to detect a dog on the image.

The model is then saved and used to classify dog images sent to telegram bot.

### Prerequisites

Python 3.
Above-mentioned [dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip) and [Xception pre-trained model bottleneck features](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogXceptionData.npz).

### Installing

```
pip install -r requirements.txt
```
The following code will train the model and save it in saved_models directory:

```
python src/train.py
```
To run the bot, insert telegram bot token into telebot_handler and run it:

```
python src/telebot_handler.py
```
### TODO

- Data augmentation
