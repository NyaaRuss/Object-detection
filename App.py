import streamlit as st
import cv2
import tensorflow as tf
import numpy as np
import sys
from keras.models import load_model

# Loading the Inception model
model = load_model('mod.h5', compile=(False))


# Functions
def predict(frame, model):
    # Pre-process the image for model prediction
    img = cv2.resize(frame, (299, 299))
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)

    img /= 255.0

    # Predict with the Inceptionv3 model
    prediction = model.predict(img)

    # Convert the prediction into text
    pred_text = tf.keras.applications.inception_v3.decode_predictions(
        prediction, top=1)
    for (i, (imagenetID, label, prob)) in enumerate(pred_text[0]):
        label = ("{}: {:.2f}%".format(label, prob * 100))

    st.markdown(label)


def predict2(frame, model):
    # Pre-process the image for model prediction
    img = cv2.resize(frame, (299, 299))
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)

    img /= 255.0

    # Predict with the Inceptionv3 model
    prediction = model.predict(img)

    # Convert the prediction into text
    pred_text = tf.keras.applications.inception_v3.decode_predictions(
        prediction, top=1)
    for (i, (imagenetID, label, prob)) in enumerate(pred_text[0]):
        pred_class = label

    return pred_class


def object_detection(search_key, frame, model):
    label = predict2(frame, model)
    label = label.lower()
    if label.find(search_key) > -1:
        st.image(frame, caption=label)

        # return sys.exit()
    else:
        if label.find == None:
            st.text('Not Found')


# Main App
def main():
    """Deployment using Streamlit"""
    st.title("Object Detection")
    st.text("Built with Streamlit and Inceptionv3")
    st.title("Instructions to be followed")
    st.text("1. Upload the video first")
    st.text("2. Detect")
    st.text("3. Then search any object in the video")

    activities = ["Detect Objects", "About"]
    choice = st.sidebar.selectbox("Choose Activity", activities)

    if choice == "Detect Objects":
        st.subheader("Upload Video")

        video_file = st.file_uploader("Choose a video...", type=["mp4", "avi"])

        if video_file is not None:
            path = video_file.name
            with open(path, mode='wb') as f:
                f.write(video_file.read())
                st.success("Saved File")
                video_file = open(path, "rb").read()
                st.video(video_file)
            cap = cv2.VideoCapture(path)
            frame_width = int(cap.get(3))
            frame_height = int(cap.get(4))

            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            output = cv2.VideoWriter(
                'output.mp4', fourcc, 20.0, (frame_width, frame_height))

            if st.button("Detect Objects"):

                # Start the video prediction loop
                while cap.isOpened():
                    ret, frame = cap.read()

                    if not ret:
                        break

                    # Perform object detection
                    predict(frame, model)

                    # Display the resulting frame

                cap.release()
                output.release()
              

            key = st.text_input('Search key')
            key = key.lower()

            if key is not None:

                if st.button("Search for an object"):

                    # Start the video prediction loop
                    while cap.isOpened():
                        ret, frame = cap.read()

                        if not ret:
                            break

                        # Perform object detection
                        object_detection(key, frame, model)
                        #frame = predict(frame, model)

                        # Display the resulting frame
                    cap.release()
                    output.release()
                    cv2.destroyAllWindows()

    elif choice == "About":
        st.subheader('Author: Nyasha R Zhou')
        st.text("R204449M")


if __name__ == '__main__':
    main()
