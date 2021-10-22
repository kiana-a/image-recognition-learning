from keras.datasets import cifar10
import matplotlib.pyplot as plt
import tensorflow

cifar10_class_names = {
    0: "Plane",
    1: "Car",
    2: "Bird",
    3: "Cat",
    4: "Deer",
    5: "Dog",
    6: "Frog",
    7: "Horse",
    8: "Boat",
    9: "Truck"
}

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

for i in range(1000):
    sample_image = x_train[i]
    image_class_number = y_train[i][0]
    image_class_name = cifar10_class_names[image_class_number]

    plt.imshow(sample_image)
    plt.title(image_class_name)
    plt.show()
