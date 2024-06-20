# -*- coding: utf-8 -*-
"""app.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1p0NjbHgp9CBfggVc30WvgITVLJ1OyM_k
"""

pip install streamlit pillow tensorflow

# NST.py
import tensorflow as tf
from PIL import Image
import numpy as np

class NST:
    def __init__(self, content_path, style_path):
        self.content_path = content_path
        self.style_path = style_path

def load_image(self, path):
        max_dim = 512
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        shape = tf.cast(tf.shape(img)[:-1], tf.float32)
        long_dim = max(shape)
        scale = max_dim / long_dim
        new_shape = tf.cast(shape * scale, tf.int32)
        img = tf.image.resize(img, new_shape)
        img = img[tf.newaxis, :]
        return img

def tensor_to_image(self, tensor):
        tensor = tensor * 255
        tensor = np.array(tensor, dtype=np.uint8)
        if np.ndim(tensor) > 3:
            assert tensor.shape[0] == 1
            tensor = tensor[0]
        return Image.fromarray(tensor)

def run(self):
        content_image = self.load_image(self.content_path)
        style_image = self.load_image(self.style_path)

        # Implement the style transfer logic here

        stylized_image = content_image  # Placeholder for the resulting image
        stylized_image = self.tensor_to_image(stylized_image)
        stylized_image.save("stylized_image.png")

# Application title and description
st.title("Neural Style Transfer")
st.markdown("""
Welcome to the **Neural Style Transfer** app! Transform your photos into artistic masterpieces by merging the content of one image with the style of another using deep learning techniques.
Upload a **content image** and a **style image**, then click the 'Style' button to create your unique artwork!
""")

def save_file(uploaded_file, file_path):
    try:
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return True
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return False

# Upload content image
content_image = st.file_uploader("Upload Content Image", type=["png", "jpg", "jpeg"])
if content_image:
    if save_file(content_image, "content_image.png"):
        st.image(content_image, caption="Content Image", use_column_width=True)

# Upload style image
style_image = st.file_uploader("Upload Style Image", type=["png", "jpg", "jpeg"])
if style_image:
    if save_file(style_image, "style_image.png"):
        st.image(style_image, caption="Style Image", use_column_width=True)

# Perform Neural Style Transfer when the 'Style' button is clicked
if st.button("Style"):
    st.write("Styling in progress... Please wait for 5-10 minutes.")
    NST('content_image.png', 'style_image.png')

if os.path.exists("stylized_image.png"):
        styled_image = Image.open("stylized_image.png")
        st.image(styled_image, caption="Styled Image", use_column_width=True)

with open("stylized_image.png", "rb") as file:
            st.download_button(
                label="Download Styled Image",
                data=file,
                file_name="stylized_image.png",
                mime="image/png"
            )

