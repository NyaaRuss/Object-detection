
import streamlit as st
import cv2
import tensorflow as tf 
import numpy as np
from keras.models import load_model
from PIL import Image
#Loading the Inception model
model= load_model('mod.h5',compile=(False))

#Functions
def splitting(name):
    vidcap = cv2.VideoCapture(name)
    success,img = vidcap.read()
    count = 0
    frame_skip=300
    while success:  
        success,img = vidcap.read()
        cv2.imwrite("Images/frame%d.jpg" % count, img)
        if count % frame_skip ==0:
            print('frame: {}'.format(count))
            pil_img= Image.fromarray(img)
            a=st.image(pil_img)
            #model.predict(a)
        count+=1
        
    preprocessing()

def preprocessing():
    x = tf.io.read_file(a)
    x = tf.io.decode_image(x,channels=3) 
    x = tf.image.resize(x,[299,299])
    x = tf.expand_dims(x, axis=0)
    x = tf.keras.applications.inception_v3.preprocess_input(x)
    return x
    
def predict(x):
    P = tf.keras.applications.inception_v3.decode_predictions(model.predict(x), top=1)
    return P
    
def main():
    st.title("Computer Vision,Deep Learning Model.")
    
    file = st.file_uploader("Upload video",type=(['mp4']))
    if file is not None:
        path = file.name
        with open(path,mode='wb') as f: 
          f.write(file.read())         
        st.success("Saved File")
        
        video_file = open(path, "rb").read()

        st.video(video_file)
        
    if st.button("Detect"):
        output1 = splitting(path)
        output2 = preprocessing()
        output = predict(output2)
    
        st.success('The Output is {}'.format(output))
        #st.success(output)

        
if __name__=='__main__':

    main()
