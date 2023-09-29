import streamlit as st
import numpy as np
import cv2
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences

# loading
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
  
model = load_model('encoderdecodermodel.h5')
modelvgg = VGG16(include_top=True, weights='imagenet')
image_model = tf.keras.Model(inputs=modelvgg.inputs, outputs=modelvgg.layers[-2].output)


def extract_frames(video_path):
    # Code to extract frames from video
    frames = []
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    return frames

def extract_image_features(frame):
    frame = cv2.resize(frame, (224, 224))
    frame = img_to_array(frame)
    frame = np.expand_dims(frame, axis=0)
    frame = frame / 255.0
    return frame

def generate_caption(frame_feature):
    in_text = '<start>'
    max_length = 30
    for _ in range(max_length):
        sequence = [word_to_idx[word] for word in in_text.split() if word in word_to_idx]
        sequence = pad_sequences([sequence], maxlen=max_length)
        prediction = model.predict([np.array([frame_feature]), np.array(sequence)])[0]
        prediction = np.argmax(prediction)
        word = idx_to_word[prediction]
        in_text += ' ' + word
        if word == '<end>':
            break
    return in_text

def generate_video_description(video_path):
    frames = extract_frames(video_path)
    frame_features = np.array([extract_image_features(frame) for frame in frames])
    descriptions = []
    for frame_feature in frame_features:
        frame_feature = np.expand_dims(frame_feature, axis=0)
        caption = generate_caption(frame_feature)
        descriptions.append(caption)
    video_caption = " ".join(descriptions)
    return video_caption
# Streamlit app


def main():
    st.title("Video Captions Generator")
    video_file = st.file_uploader("Upload a video", type=['mp4', 'avi'], accept_multiple_files=False, key="video_uploader")


    if video_file is not None:
        # Save video file
        video_path = 'uploaded_video.mp4'
        with open(video_path, 'wb') as f:
            f.write(video_file.read())

        # Generate video description
        video_description = generate_video_description(video_path)

        # Display video description
        st.subheader("The generated captions:")
        st.write(video_description)

if __name__ == '__main__':
    main()
