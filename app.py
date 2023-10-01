import streamlit as st
import cv2
import numpy as np
from tensorflow import keras
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# Set the maximum allowed file size to 2MB
st.set_max_upload_size(2 * 1024 * 1024)

model = load_model('encoderdecodermodel.h5')


def preprocess_image(image):
    image = cv2.resize(image, (299, 299))
    image = img_to_array(image)
    image = preprocess_input(image)
    return image

def generate_description(model, frames):
    description = ''
    for frame in frames:
        frame = preprocess_image(frame)
        frame = np.expand_dims(frame, axis=0)
        features = encoder_model.predict(frame)
        input_seq = 'start'
        while True:
            sequence = [word_to_idx[word] for word in input_seq.split() if word in word_to_idx]
            sequence = pad_sequences([sequence], maxlen=max_length)
            yhat = decoder_model.predict([features, sequence], verbose=0)
            yhat = np.argmax(yhat)
            word = idx_to_word[yhat]
            input_seq += ' ' + word
            if word == 'end':
                break
            description += ' ' + word
    return description.strip()

def main():
    st.title("Video Description Generator")

    # Load encoder-decoder model
    encoder_model, decoder_model = load_encoder_decoder_model()

    # Upload video file
    video_file = st.file_uploader("Upload a video (Max size: 2MB)", type=["mp4", "avi", "mov"])

    if video_file is not None:
        # Read the video file
        video = cv2.VideoCapture(video_file)
        frames = []

        # Extract frames from the video
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break
            frames.append(frame)

        video.release()

        # Generate description using the encoder-decoder model
        description = generate_description(model, frames)

        # Display the generated description
        st.subheader("Video Description")
        st.write(description)

if __name__ == '__main__':
    main()
