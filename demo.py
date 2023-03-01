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

with open("utils/knn_model.pkl", 'rb') as file:
    knn_loaded = pickle.load(file)

detector = cv2.CascadeClassifier("utils/haarcascade_frontalface_default.xml")


N = 6480
orientations = 5
pixels_per_cell = (16, 16)
cells_per_block = (4, 4)


LIST_FACES = []

def inference_choose_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    rects = detector.detectMultiScale(gray, scaleFactor=1.05,
    minNeighbors=15, minSize=(50, 50),
    flags=cv2.CASCADE_SCALE_IMAGE)

    # loop over the bounding boxes

    image = img.copy()

    global LIST_FACES
    LIST_FACES = []

    global RECTS
    RECTS = rects

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


def inference_predict_age(img, face_nums):

    print("inference_predict_age")
    print(face_nums)
    list_faces = [LIST_FACES[i-1] for i in face_nums]
    rects = [RECTS[i-1] for i in face_nums]

    image = img.copy()

    for coord, face, index in zip(rects, list_faces, face_nums):
        (x, y, w, h) = coord
        face_image = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
        face_image_gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        fd = hog(face_image_gray, orientations=orientations, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block, block_norm='L2', visualize=False, transform_sqrt=True)
        predict = knn_loaded.predict([fd])
        predict = predict[0]
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, "#{}-pred: {}".format(index, predict), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return image, gr.Dropdown.update(choices=[str(i) for i in face_nums])


def inference_see_detail(img, see_detail):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    print("inference_see_detail")
    print(see_detail)
    return img


demo = gr.Blocks()

options = []


with demo:
    img_input = gr.Image(label='input')
    
    b1 = gr.Button("detect all faces")

    img_input_1 = gr.Image(label='face prediction')
    face_nums = gr.Dropdown([], value=[], multiselect=True)
    
    b2 = gr.Button("predict age")
    
    img_predict = gr.Image(label='img face prediction')

    see_detail_option = gr.Dropdown(options, label="options to see detail")

    b3 = gr.Button("see detail")

    img_detail = gr.Image(label='img_detail')

    b1.click(inference_choose_face, inputs=[img_input], outputs=[img_input_1, face_nums])
    b2.click(inference_predict_age, inputs=[img_input, face_nums], outputs=[img_predict, see_detail_option])
    b3.click(inference_see_detail, inputs=[img_input, see_detail_option], outputs=img_detail)

demo.launch(debug=True)