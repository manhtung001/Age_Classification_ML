import os
import numpy as np
import cv2
import argparse
import random
import gradio as gr
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pickle
from skimage.feature import hog
import matplotlib.pyplot as plt
import pandas as pd
import io
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import Image
# import imutils
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D, AveragePooling2D, GlobalAveragePooling2D

face_detect_prototxt = 'utils/deploy.prototxt'
face_detect_model = 'utils/res10_300x300_ssd_iter_140000.caffemodel'
face_detect_net = cv2.dnn.readNetFromCaffe(face_detect_prototxt, face_detect_model)


with open("utils/knn_model.pkl", 'rb') as file:
    knn_loaded = pickle.load(file)

detector = cv2.CascadeClassifier("utils/haarcascade_frontalface_default.xml")

train_df = pd.read_csv("utils/train_df.csv")


# Defining the architecture of the sequential neural network.

num_classes = 7
final_age_cnn = Sequential()

# Input layer with 32 filters, followed by an AveragePooling2D layer.
final_age_cnn.add(Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=(200, 200, 1)))    # 3rd dim = 1 for grayscale images.
final_age_cnn.add(AveragePooling2D(pool_size=(2,2)))

# Three Conv2D layers with filters increasing by a factor of 2 for every successive Conv2D layer.
final_age_cnn.add(Conv2D(filters=64, kernel_size=3, activation='relu'))
final_age_cnn.add(AveragePooling2D(pool_size=(2,2)))

final_age_cnn.add(Conv2D(filters=128, kernel_size=3, activation='relu'))
final_age_cnn.add(AveragePooling2D(pool_size=(2,2)))

final_age_cnn.add(Conv2D(filters=256, kernel_size=3, activation='relu'))
final_age_cnn.add(AveragePooling2D(pool_size=(2,2)))

# A GlobalAveragePooling2D layer before going into Dense layers below.
# GlobalAveragePooling2D layer gives no. of outputs equal to no. of filters in last Conv2D layer above (256).
final_age_cnn.add(GlobalAveragePooling2D())

# One Dense layer with 132 nodes so as to taper down the no. of nodes from no. of outputs of GlobalAveragePooling2D layer above towards no. of nodes in output layer below (7).
final_age_cnn.add(Dense(132, activation='relu'))

# Output layer with 7 nodes (equal to the no. of classes).
final_age_cnn.add(Dense(num_classes, activation='softmax'))

final_age_cnn.summary()
final_age_cnn.load_weights('utils/cnn_model_checkpoint_1.h5')
# final_age_cnn.load_weights('utils/cnn_model_checkpoint_2.h5')

N = 6480
orientations = 5
pixels_per_cell = (16, 16)
cells_per_block = (4, 4)


LIST_FACES = []
LIST_EMB_FACES = {}

OPTION_DETECT_FACES = ''

# def predict_age_detection(image_file):
#     image_string = tf.io.read_file(image_file)
#     image_decoded = tf.io.decode_jpeg(image_string, channels=3)
#     image_decoded = tf.image.resize(image_decoded, [200, 200], method=tf.image.ResizeMethod.BILINEAR)
#     image_decoded = tf.reshape(image_decoded, [1, 200, 200, 3])

#     final_cnn_pred = final_age_cnn.predict(image_decoded)
#     final_cnn_pred = final_cnn_pred.argmax(axis=-1)
#     return final_cnn_pred[0]


def inference_choose_face_1(img):

    print("inference_choose_face_1")
    print(OPTION_DETECT_FACES)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    rects = detector.detectMultiScale(gray, scaleFactor=1.05,
    minNeighbors=7, minSize=(50, 50),
    flags=cv2.CASCADE_SCALE_IMAGE)

    # loop over the bounding boxes

    image = img.copy()

    global LIST_FACES
    LIST_FACES = []

    global RECTS
    RECTS = rects

    global LIST_EMB_FACES
    LIST_EMB_FACES = {}

    i = 0

    for (x, y, w, h) in rects:
        # draw the face bounding box on the image and number it
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, "#{}".format(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        face_image = image[y:y+h, x:x+w]
        face_image = cv2.resize(face_image, (200, 200))
        LIST_FACES.append(face_image)
        i += 1
        
    num_boxes = [(i+1) for i in range(len(rects))]

    return image, num_boxes


def inference_choose_face_2(img):

    # convert to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    image = img.copy()
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # save img
    # cv2.imwrite("img_via.jpg", image)
    
    print("inference_choose_face_2")
    print(OPTION_DETECT_FACES)

    (h_img, w_img) = image.shape[:2]
    print((h_img, w_img))
    blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_detect_net.setInput(blob)
    detections = face_detect_net.forward()

    rects = []

    i = 0

    global LIST_FACES
    LIST_FACES = []

    global LIST_EMB_FACES
    LIST_EMB_FACES = {}

    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the prediction
        confidence = detections[0, 0, i, 2]
        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence threshold
        if confidence > 0.5:
            # compute the (x, y)-coordinates of the bounding box for the object
            box = detections[0, 0, i, 3:7] * np.array([w_img, h_img, w_img, h_img])
            (startX, startY, endX, endY) = box.astype("int")
            x = startX
            y = startY
            w = endX - startX
            h = endY - startY
            rects.append((x, y, w, h))
            # draw the face bounding box on the image and number it
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # cv2.putText(image, "#{}".format(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(image, "#{} - {:.2f}%".format(i + 1, confidence * 100), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            face_image = image[y:y+h, x:x+w]
            face_image = cv2.resize(face_image, (200, 200))
            LIST_FACES.append(face_image)
            # i += 1

    global RECTS
    RECTS = rects

    num_boxes = [(i+1) for i in range(len(rects))]

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    return image, num_boxes


def inference_predict_age_1(img, face_nums):

    print("inference_predict_age_1")
    print(face_nums)
    list_faces = [LIST_FACES[i-1] for i in face_nums]
    rects = [RECTS[i-1] for i in face_nums]

    image = img.copy()

    for coord, face, index in zip(rects, list_faces, face_nums):
        (x, y, w, h) = coord
        face_image = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
        face_image_gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        fd = hog(face_image_gray, orientations=orientations, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block, block_norm='L2', visualize=False, transform_sqrt=True)
        LIST_EMB_FACES[index] = fd
        predict = knn_loaded.predict([fd])
        predict = predict[0]
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, "#{}-pred: {}".format(index, predict), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return image, gr.Dropdown.update(choices=[str(i) for i in face_nums])


def inference_predict_age_2(img, face_nums):
    
    print("inference_predict_age_2")
    print(face_nums)
    list_faces = [LIST_FACES[i-1] for i in face_nums]
    rects = [RECTS[i-1] for i in face_nums]

    image = img.copy()

    for coord, face, index in zip(rects, list_faces, face_nums):
        (x, y, w, h) = coord
        face_image = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
        # face_image = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        # face_image = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
        # face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        face_image = cv2.resize(face_image, (200, 200))
        face_image = tf.convert_to_tensor(face_image)
        face_image = tf.reshape(face_image, [1, 200, 200, 1])
        final_cnn_pred = final_age_cnn.predict(face_image)
        predict = final_cnn_pred.argmax(axis=-1)
        predict = predict[0]
        if 0 <= predict <= 2:
            predict = 0
        elif 3 <= predict <= 4:
            predict = 1
        else:
            predict = 2
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, "#{}-pred: {}".format(index, predict), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return image, gr.Dropdown.update(choices=[str(i) for i in face_nums])


def inference_see_detail(img_org, see_detail, option_age_predict):
    
    if img_org is None:
        raise gr.Error("Please upload an image first!")
    if option_age_predict == "via_DL":
        raise gr.Error("via_DL not support this function!")
    
    print("inference_see_detail")
    print(see_detail)
    print(LIST_EMB_FACES)

    fd = LIST_EMB_FACES[int(see_detail)]

    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    distances, indices = knn_loaded.kneighbors([fd], return_distance = True)

    if len(indices) > 0:
        distances = distances[0]
        indices = indices[0]

    # Plot the nearest neighbors
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

    i = 0
    for indice, distance in zip(indices, distances):
        path_img = "utils/combined_faces/" + train_df.iloc[indice]["filename"]
        print(path_img)
        img = cv2.imread(path_img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        axs[i].imshow(img)
        axs[i].set_title(f"age: {train_df.iloc[indice]['age']}, distance: {distance}\n{train_df.iloc[indice]['filename']}")
        i+=1

    # Convert figure to PIL image
    canvas = FigureCanvas(fig)
    buf = io.BytesIO()
    canvas.print_png(buf)
    buf.seek(0)
    img = Image.open(buf)

    return img

demo = gr.Blocks()

options = []

def handle_choose_option_face_detect(img_input, option_detect_face):
    
    if img_input is None:
        raise gr.Error("Please upload an image first!")
    
    if option_detect_face == "via_haar":
        print("via_haar")
        return inference_choose_face_1(img_input)
    elif option_detect_face == "via_resnet":
        print("via_resnet")
        return inference_choose_face_2(img_input)
    

def handle_choose_option_face_predict(img_input, face_nums, option_age_predict):
    
    if img_input is None:
        raise gr.Error("Please upload an image first!")
    
    if option_age_predict == "via_ML":
        print("via_ML")
        return inference_predict_age_1(img_input, face_nums)
    elif option_age_predict == "via_DL":
        print("via_DL")
        return inference_predict_age_2(img_input, face_nums)



with demo:
    img_input = gr.Image(label='input')

    option_detect_face = gr.Radio(["via_haar", "via_resnet"], label="kind_detect", info="Kind detect face", value="via_haar")
    
    b1 = gr.Button("detect all faces")

    img_input_1 = gr.Image(label='face prediction')
    face_nums = gr.Dropdown([], value=[], multiselect=True)
    
    option_age_predict = gr.Radio(["via_ML", "via_DL"], label="kind_predict", info="Kind predict face", value="via_ML")
    
    b2 = gr.Button("predict age")
    
    img_predict = gr.Image(label='img face prediction')

    see_detail_option = gr.Dropdown(options, label="options to see detail")

    b3 = gr.Button("see detail")

    img_detail = gr.Image(label='img_detail')

    b1.click(handle_choose_option_face_detect, inputs=[img_input, option_detect_face], outputs=[img_input_1, face_nums])
    b2.click(handle_choose_option_face_predict, inputs=[img_input, face_nums, option_age_predict], outputs=[img_predict, see_detail_option])
    b3.click(inference_see_detail, inputs=[img_input, see_detail_option, option_age_predict], outputs=img_detail)

demo.launch(debug=True)




