from PIL import Image
import streamlit as st
import cv2
import tensorflow as tf 
import numpy as np
from keras.models import load_model
from PIL import Image
import PIL

#Loading the Inception model
model= load_model('frames.h5',compile=(False))

def splitting(name):
    vidcap = cv2.VideoCapture(name)
    success,frame = vidcap.read()
    count = 0
    frame_skip =1
    while success:
        success, frame = vidcap.read() # get next frame from video
        cv2.imwrite(r"img/frame%d.jpg" % count, frame) 
        if count % frame_skip == 0: # only analyze every n=300 frames
            #print('frame: {}'.format(count)) 
            pil_img = Image.fromarray(frame) # convert opencv frame (with type()==numpy) into PIL Image
            #st.image(pil_img)
        if count > 20 :
            break
        count += 1
    preprocessing()

def main():
    
    st.title("Image Captioning")

    file = st.file_uploader("Upload video",type=(['mp4']))
    if file is not None: # run only when user uploads video
        vid = file.name
        with open(vid, mode='wb') as f:
            f.write(file.read()) # save video to disk

        st.markdown(f"""
        ### Files
        - {vid}
        """,
        unsafe_allow_html=True) # display file name

        vidcap = cv2.VideoCapture(vid) # load video from disk
        cur_frame = 0
        success = True

def caption():
    
    original = Image.open(image_data)
    original = original.resize((224, 224), Image.ANTIALIAS)
    numpy_image = img_to_array(original)
    nimage = preprocess_input(numpy_image)
    
    feature = modelvgg.predict(nimage.reshape( (1,) + nimage.shape[:3]))
    caption = predict_caption(feature)
    table = st.Label(frame, text="Caption: " + caption[9:-7], font=("Helvetica", 12)).pack()

root.title('IMAGE CAPTION GENERATOR')
root.iconbitmap('class.ico')
root.resizable(False, False)
tit = st.Label(root, text="IMAGE CAPTION GENERATOR", padx=25, pady=6, font=("", 12)).pack()
canvas = st.Canvas(root, height=550, width=600, bg='#D1EDf2')
canvas.pack()
frame = st.Frame(root, bg='white')
frame.place(relwidth=0.8, relheight=0.8, relx=0.1, rely=0.1)
chose_image = st.Button(root, text='Choose Image',
                        padx=35, pady=10,
                        fg="black", bg="pink", command=upload_img, activebackground="#add8e6")
chose_image.pack(side=tk.LEFT)

caption_image = st.Button(root, text='Classify Image',
                        padx=35, pady=10,
                        fg="black", bg="pink", command=caption, activebackground="#add8e6")
caption_image.pack(side=tk.RIGHT)
root.mainloop()
