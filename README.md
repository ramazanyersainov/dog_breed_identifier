# Dog Breed Identification Bot

Image classification using transfer learning. Based on [Udacity's dog dataset and and Xception pre-trained model](https://github.com/udacity/dog-project). 

The first model uses Udacity's prepared bottleneck features. The model takes it as the input layer, applies regularization and droupout layers, and outputs in 133 different classes (dog breeds). Test set accuracy results in 79%.

The second model does not use these features and trains Xception model with regularization applied on it from the dataset itself. Data augmentation was applied on training set. Additionaly, I have added my a set of 100 photos of my dog, Sunny, as the separate breed class. Test set accuracy results in 83% (10 epochs).

Before passing data to the breed identification model, it uses ResNet50 model to detect a dog on the image.

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
