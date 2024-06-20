# Neural-Style-Transfer
This project implements Neural Style Transfer using the VGG19 model with pre-trained ImageNet weights. Neural Style Transfer is a technique that takes two images—a content image and a style image—and blends them together so the output image looks like the content image painted in the style of the style image.
Libraries Used

tensorflow: TensorFlow is an open-source library for machine learning. It is used here for building and training the neural network model.
numpy : NumPy is a fundamental package for scientific computing in Python. It is used for handling arrays and performing numerical operations.
PIL : The Python Imaging Library (PIL) is used for opening, manipulating, and saving many different image file formats. We use the Pillow fork of PIL.
matplotlib : Matplotlib is a plotting library for the Python programming language and its numerical mathematics extension NumPy. It is used for displaying images.
Code Desctription :-

Loading the VGG19 Model:
The VGG19 model is loaded with pre-trained ImageNet weights. This model is used to extract features from the content and style images.
Loading and Processing Images:
The content and style images are loaded and processed for input into the neural network.
Training Setup:
An optimizer (Adam) is used for training the neural network. The train_step function is defined to perform a single step of training, computing the loss and applying gradients.
Training Loop:
The model is trained for a specified number of epochs, and the progress is displayed. The output image is displayed at the end of each epoch.
Saving and Displaying the Result:
The final stylized image is saved and displayed.
Usage

Install the required libraries:
Place your content and style images in the working directory and set their paths in the code:
Run the script:
python your_script_name.py
The output stylized image will be saved as stylized_image.png in the working directory.
References :
TensorFlow
NumPy
Pillow (PIL Fork)
Matplotlib
