# Neural-Style-Transfer
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time
def NST(content_img, style_img):

    # Load the VGG19 model pre-trained on ImageNet, excluding the top fully connected layers
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
# Define the layers used for content and style extraction
content_layers = ['block5_conv2']
style_layers = [
        'block1_conv1', 'block2_conv1', 'block3_conv1',
        'block4_conv1', 'block5_conv1'
    ]
num_content_layers = len(content_layers)
num_style_layers = len(style_layers)
def load_img(image_path):
        max_dim = 512
        img = tf.io.read_file(image_path)
        img = tf.image.decode_image(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)

        shape = tf.cast(tf.shape(img)[:-1], tf.float32)
        long_dim = max(shape)
        scale = max_dim / long_dim

        new_shape = tf.cast(shape * scale, tf.int32)
        img = tf.image.resize(img, new_shape)
        img = img[tf.newaxis, :]
        return img
def vgg_layers(layer_names):
        vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
        vgg.trainable = False
        outputs = [vgg.get_layer(name).output for name in layer_names]
        return tf.keras.Model([vgg.input], outputs)
        
def gram_matrix(input_tensor):
        result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
        input_shape = tf.shape(input_tensor)
        num_elements = tf.cast(input_shape[1] * input_shape[2], tf.float32)
        return result / num_elements
class StyleContentModel(tf.keras.models.Model):
    def __init__(self, style_layers, content_layers):
        super(StyleContentModel, self).__init__()
        self.vgg =  vgg19.VGG19(include_top=False, weights='imagenet')
        self.vgg.trainable = False
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg_layers_dict = dict([(layer.name, layer.output) for layer in self.vgg.layers])
        self.style_outputs = [self.vgg_layers_dict[name] for name in style_layers]
        self.content_outputs = [self.vgg_layers_dict[name] for name in content_layers]
    def call(self, inputs): # Add this call method
        """
        This method takes the input image and returns the style and content representations.
        """
        inputs = inputs * 255.0
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(preprocessed_input)
        style_outputs = [style_layer[0] for style_layer in outputs[:self.num_style_layers]]
        content_outputs = [content_layer[0] for content_layer in outputs[self.num_style_layers:]]
        return {'content': content_outputs, 'style': style_outputs}
def compute_loss(outputs, style_targets, content_targets):
        style_outputs = outputs['style']
        content_outputs = outputs['content']
        style_loss = tf.add_n([tf.reduce_mean((style_outputs[name] - style_targets[name])**2) for name in style_outputs.keys()])
        style_loss *= 1e-2 / num_style_layers

        content_loss = tf.add_n([tf.reduce_mean((content_outputs[name] - content_targets[name])**2) for name in content_outputs.keys()])
        content_loss *= 1e5 / num_content_layers

        total_loss = style_loss + content_loss
        return total_loss
# Define paths to your content and style images
content_img = '/content/Logo.png'
style_img = '/content/Frame 34.png'

# Load content image
content_image = load_img(content_img)

# Load style image
style_image = load_img(style_img)
def extract_targets(style_image_path, content_image_path, style_layers, content_layers):
    # Load style and content images
    style_image = tf.keras.preprocessing.image.load_img(style_image_path)
    content_image = tf.keras.preprocessing.image.load_img(content_image_path)

    # Convert images to TensorFlow tensors
    style_image = tf.keras.preprocessing.image.img_to_array(style_image)
    content_image = tf.keras.preprocessing.image.img_to_array(content_image)

    # Expand dimensions to match batch size (1 in this case)
    style_image = tf.expand_dims(style_image, axis=0)
    content_image = tf.expand_dims(content_image, axis=0)

    # Instantiate StyleContentModel
    extractor = StyleContentModel(style_layers, content_layers)

    # Get style and content targets
    style_targets = extractor(style_image)['style']
    content_targets = extractor(content_image)['content']

    return style_targets, content_targets
optimizer = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)
style_image_path = '/content/style_img_page-0001.jpg'
content_image_path = '/content/content_img_page-0001.jpg'

# Define your style and content layers
style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1']
content_layers = ['block5_conv2']
# add function definitions for yellow lines

@tf.function()
def train_step(image):
   with tf.GradientTape() as tape:
        outputs = model(image)
        loss = compute_loss(outputs, style_targets, content_targets)
        grad = tape.gradient(loss, image)
        optimizer.apply_gradients([(grad, image)])
        image.assign(tf.clip_by_value(image, 0.0, 1.0))

generated_image = tf.Variable(content_image)
def tensor_to_image(tensor):
        tensor = tensor * 255
        tensor = np.array(tensor, dtype=np.uint8)
        if np.ndim(tensor) > 3:
            assert tensor.shape[0] == 1
            tensor = tensor[0]
        return Image.fromarray(tensor)
start_time = time.time()
epochs = 5
steps_per_epoch = 100
step = 0
losses = []

for epoch in range(epochs):
    for step in range(steps_per_epoch):
        step += 1
        with tf.device('/gpu:0'):
          loss = train_step(generated_image)
        losses.append(loss.numpy())
        print(".", end='', flush=True)

    end_time = time.time()
    print(f"\nEpoch {epoch+1}/{epochs} - Time: {end_time - start_time:.2f}s")
    start_time = time.time()

    # Define loss for current epoch
    loss = train_step(generated_image)

    print("Loss: {:.2f}".format(loss.numpy()))

# Plot the loss convergence
plt.figure(figsize=(10, 6))
plt.plot(losses, label='Loss')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Loss convergence over training iterations')
plt.legend()
plt.show()

result_image = tensor_to_image(generated_image)
result_image.save('stylized_image.png')







        
