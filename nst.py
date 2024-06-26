# -*- coding: utf-8 -*-
"""NST.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1kvgohXQ_h7xW7v97DmcfY5KWjwujlnPb
"""

import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load the VGG19 model with pre-trained ImageNet weights
vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')

# Define the layers for content and style extraction
content_layers = ['block5_conv2']
style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)

def load_image(image_path):
    max_dim = 512
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)

    shape = tf.cast(tf.shape(image)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)
    image = tf.image.resize(image, new_shape)
    image = image[tf.newaxis, :]
    return image

def numpy_to_tensor(array):
    return tf.convert_to_tensor(array, dtype=tf.float32)

def build_vgg_model(layer_names):
    vgg.trainable = False
    outputs = [vgg.get_layer(name).output for name in layer_names]
    return tf.keras.Model([vgg.input], outputs)

def compute_gram_matrix(tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', tensor, tensor)
    input_shape = tf.shape(tensor)
    num_elements = tf.cast(input_shape[1] * input_shape[2], tf.float32)
    return result / num_elements

class StyleContentModel(tf.keras.models.Model):
    def __init__(self, style_layers, content_layers):
        super(StyleContentModel, self).__init__()
        self.vgg = build_vgg_model(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False

    def call(self, inputs):
        #instead of inputs *= 255.0, just modify the input directly, create a new tensor
        preprocessed_input = inputs * 255.0
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(preprocessed_input)
        outputs = self.vgg(preprocessed_input)
        style_outputs, content_outputs = outputs[:self.num_style_layers], outputs[self.num_style_layers:]

        style_outputs = [compute_gram_matrix(output) for output in style_outputs]

        content_dict = {content_name: value for content_name, value in zip(self.content_layers, content_outputs)}
        style_dict = {style_name: value for style_name, value in zip(self.style_layers, style_outputs)}

        return {'content': content_dict, 'style': style_dict}

def calculate_loss(outputs, style_targets, content_targets):
    style_outputs = outputs['style']
    content_outputs = outputs['content']

    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name] - style_targets[name])**2) for name in style_outputs.keys()])
    style_loss *= 1e-2 / num_style_layers

    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name] - content_targets[name])**2) for name in content_outputs.keys()])
    content_loss *= 1e5 / num_content_layers

    total_loss = style_loss + content_loss
    return total_loss

# Example paths for content and style images
content_img_path = '/content_img_page-0001.jpg'
style_img_path = '/style_img_page-0001.jpg'

# Verify the path
print("Content image path:", content_img_path)

# Check if the file exists (replace with appropriate shell command for your environment)
!ls /content_img_page-0001.jpg # Example for Linux

# Load and process the images
content_image = load_image(content_img_path)
style_image = load_image(style_img_path)

# Create an instance of the StyleContentModel
model = StyleContentModel(style_layers, content_layers)

# Now you can use the model
style_targets = model(style_image)['style']
content_targets = model(content_image)['content']

# Convert the images from numpy arrays to tensors if needed
# If the images are already in tensor format, you can skip this step
# content_image = numpy_to_tensor(content_img_array)
# style_image = numpy_to_tensor(style_img_array)

model = StyleContentModel(style_layers, content_layers)

optimizer = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

@tf.function
def train_step(image):
    with tf.GradientTape() as tape:
        outputs = model(image)
        loss = calculate_loss(outputs, style_targets, content_targets)
    grad = tape.gradient(loss, image)
    optimizer.apply_gradients([(grad, image)])
    image.assign(tf.clip_by_value(image, 0.0, 1.0))

image = tf.Variable(content_image)

def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if tensor.ndim > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return Image.fromarray(tensor)

import time

start_time = time.time()  # Start the timer

epochs = 5
steps_per_epoch = 100

for epoch in range(epochs):
    epoch_start_time = time.time()  # Start the timer for the epoch
    for step in range(steps_per_epoch):
        step_start_time = time.time()  # Start the timer for the step
        step_end_time = time.time()  # End the timer for the step
        print('.', end='', flush=True)
        print(f" Step {step+1}/{steps_per_epoch} - Time: {step_end_time - step_start_time:.2f}s", flush=True)

    epoch_end_time = time.time()  # End the timer for the epoch
    print(f"Epoch {epoch + 1}/{epochs} - Time: {epoch_end_time - epoch_start_time:.2f}s")

    plt.imshow(tensor_to_image(image))
    plt.title(f'Epoch {epoch + 1}')
    plt.axis('off')
    plt.show()

end_time = time.time()  # End the timer for the entire process
print(f"Total time: {end_time - start_time:.1f} seconds")

result_image = tensor_to_image(image)
result_image.save('stylized_image.png')

# Load the stylized image
result_image = Image.open('stylized_image.png')
result_image.show() # Display the image

# Save the stylized image to a file (e.g., PNG)
result_image.save('stylized_image.png')

# Optionally, open and display the saved image using PIL
saved_image = Image.open('stylized_image.png')
saved_image.show()

