import streamlit as st
import os
import numpy as np
import cv2
import argparse
import random
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pickle
from skimage.feature import hog
import matplotlib.pyplot as plt
import pandas as pd
import io
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import Image

st.set_page_config(
    page_title="Hệ thống dự đoán tuổi dựa trên khuân mặt",
    page_icon="📊",
    initial_sidebar_state="expanded",
)

st.header(
    """
Hệ thống dự đoán tuổi dựa trên khuân mặt
"""
)

import os
import shutil


# conda activate csdlpt
# streamlit run D:/Desktop/Age_Classification_ML/app.py


@st.cache
def reset_folder():
    print("reset_folder")
    dir_path = os.path.dirname(os.path.realpath(__file__))
    tmpPath = os.path.join(dir_path, "tmp")
    if os.path.exists(tmpPath):
        shutil.rmtree(tmpPath)
    if not os.path.exists(tmpPath):
        os.mkdir(tmpPath)

with open("D:/Desktop/Age_Classification_ML/utils/knn_model.pkl", 'rb') as file:
    knn_loaded = pickle.load(file)

detector = cv2.CascadeClassifier("D:/Desktop/Age_Classification_ML/utils/haarcascade_frontalface_default.xml")

train_df = pd.read_csv("D:/Desktop/Age_Classification_ML/utils/train_df.csv")

form1_success = False
form2_success = False

st.write("pic:")
# with st.form("form1", clear_on_submit=True):


print("root")

def form1():
    with st.form("form1", clear_on_submit=False):
        uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
        submitted = st.form_submit_button("Detect faces")

        if submitted:
            LIST_FACES = []
            if uploaded_file is not None:
                img = Image.open(uploaded_file)
                img = np.array(img)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                rects = detector.detectMultiScale(gray, scaleFactor=1.05,
                    minNeighbors=7, minSize=(50, 50),
                    flags=cv2.CASCADE_SCALE_IMAGE)
                RECTS = rects
                image = img.copy()
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
                st.image(image, caption='Detect Faces Image.', use_column_width=True)
                # Set the form 1 success flag to True
                form1_success = True
                return LIST_FACES, RECTS, num_boxes
            else:
                # show error
                st.error("Please upload an image file")
                st.stop()

# def form2(num_boxes):


# if form1_success:
#     # face_nums = st.multiselect("Choose faces", num_boxes, num_boxes)
#     face_nums = num_boxes
#     if len(face_nums) > 0:
#         # Set the form 2 success flag to True
#         form2_success = True
#     else:
#         # show error
#         st.error("Please choose at least one face")
#         st.stop()

# if form2_success:
#     with st.form("form3"):
#         submitted = st.form_submit_button("Predict")
#         if submitted:
#             print("form3")
#             print(face_nums)
#             print(LIST_FACES)
#             print(RECTS)
#             list_faces = [LIST_FACES[i-1] for i in face_nums]
#             rects = [RECTS[i-1] for i in face_nums]
#             print(list_faces)
#             print(rects)

if __name__ == "__main__":
    # list_faces, rects, num_boxes = form1()
    list_faces, rects, num_boxes = st.cache_resource(form1)
    
    num_boxes = st.multiselect("Choose faces", num_boxes, num_boxes)
    print("main")
    print(num_boxes)
