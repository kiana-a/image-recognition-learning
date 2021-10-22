from keras.models import model_from_json
from pathlib import Path
from keras.preprocessing import image
import numpy as np
from keras.applications import vgg16

f = Path("../code/model_structure.json")
model_structure = f.read_text()

model = model_from_json(model_structure)

model.load_weights("model_weights.h5")

img = image.load_img("not_dog.png", target_size=(64, 64))

image_array = image.img_to_array(img)

images = np.expand_dims(image_array, axis=0)

images = vgg16.preprocess_input(images)

feature_extraction_model = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(64, 64, 3))
features = feature_extraction_model.predict(images)

results = model.predict(features)

single_result = results[0][0]

print("Likelihood that this image contains a dog: {}%".format(int(single_result * 100)))