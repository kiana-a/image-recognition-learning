import numpy as np
from keras.preprocessing import image
from keras.applications import vgg16

model = vgg16.VGG16()

img = image.load_img("bay.jpg", target_size=(224, 224))

x = image.img_to_array(img)

x = np.expand_dims(x, axis=0)

x = vgg16.preprocess_input(x)

predictions = model.predict(x)

predicted_classes = vgg16.decode_predictions(predictions)

print("Top predictions for this image:")

for imagenet_id, name, likelihood in predicted_classes[0]:
    print("Prediction: {} - {:2f}".format(name, likelihood))

