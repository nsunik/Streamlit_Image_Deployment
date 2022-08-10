import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageOps

net = cv2.dnn.readNetFromTensorflow('tf_final.pb')

label = ['Severe Tropical Storm', 'Tropical Depression', 'Tropical Storm', 'Typhoon']

file = st.file_uploader("Please upload cyclone image", type=["jpg", "png", 'jpeg'])

if file is not None:
	file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
	opencv_image = cv2.imdecode(file_bytes, 1)
	opencv_image = cv2.cvtColor(opencv_image,cv2.COLOR_BGR2RGB)
	image = Image.open(file)
	#st.write(opencv_image.shape)
	st.image(image, use_column_width=True)

	blob = cv2.dnn.blobFromImage(opencv_image, scalefactor=1.0 / 255, size=(224, 224),mean=(0, 0, 0))
	net.setInput(blob)
	out = net.forward()
	out = out.flatten()
	classId = np.argmax(out)
	confidence = np.round(out[classId]*100,2)
	op = f'{label[classId]} - {confidence}%'
	st.write(op)
