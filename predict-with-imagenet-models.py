import numpy as np
import matplotlib.pyplot as plt
import cv2

import keras
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import decode_predictions

from keras.applications import vgg16, inception_v3, resnet50, mobilenet

models = []
names = ("VGG16", "Inception", "ResNet50", "MobileNet")
networks = (vgg16, inception_v3, resnet50, mobilenet)
labels = []
amount = 3


def load_models():
    models.append(vgg16.VGG16(weights='imagenet'))
    models.append(inception_v3.InceptionV3(weights='imagenet'))
    models.append(resnet50.ResNet50(weights='imagenet'))


def predict(image, model_index):
    processed_image = networks[model_index].preprocess_input(image.copy())
    predictions = models[model_index].predict(processed_image)
    labels.append(decode_predictions(predictions, top=5))


def print_image(imagex):
    image = np.uint8(img_to_array(imagex)).copy()
    image = cv2.resize(image, (900, 900))

    for i in range(amount):
        label = labels[i]
        cv2.putText(image, "{}: {}, {:.2f}".format(names[i], label[0][0][1], label[0][0][2]), (400, 40 + 35 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

    image = cv2.resize(image, (700, 700))
    # cv2.imwrite("{}_output.jpg".format("images/"),
    #             cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    cv2.imshow("Wynik", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    # plt.figure(figsize=[10, 10])
    # plt.imshow(image)
    # plt.axis('off')
    # plt.show()


def main():
    load_models()

    while 1:
        print("Dej sciezke lub 0")
        path = input()
        labels.clear()

        if path == "0": exit()

        try:
            original = load_img(path, target_size=(224, 224))
        except Exception as e:
            print("Wrong path\n" + str(e))

            continue
        numpy_image = img_to_array(original)
        batch_image = np.expand_dims(numpy_image, axis=0)

        for i in range(amount):
            predict(batch_image, i)

        print(labels)
        print_image(numpy_image)


if __name__ == "__main__":
    main()

    # numpy_image = np.uint8(img_to_array(original)).copy()
    # numpy_image = cv2.resize(numpy_image, (900, 900))
