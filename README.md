# Neural-Style-Transfer
This project implements Neural Style Transfer using the VGG19 model with pre-trained ImageNet weights. Neural Style Transfer is a technique that takes two images—a content image and a style image—and blends them together so the output image looks like the content image painted in the style of the style image.
Libraries Used

tensorflow: TensorFlow is an open-source library for machine learning. It is used here for building and training the neural network model.
numpy : NumPy is a fundamental package for scientific computing in Python. It is used for handling arrays and performing numerical operations.
PIL : The Python Imaging Library (PIL) is used for opening, manipulating, and saving many different image file formats. We use the Pillow fork of PIL.
matplotlib : Matplotlib is a plotting library for the Python programming language and its numerical mathematics extension NumPy. It is used for displaying images.
Code Desctription :-

Loading the VGG19 Model:
1) The VGG19 model is loaded with pre-trained ImageNet weights. This model is used to extract features from the content and style images.
2) Loading and Processing Images
3) The content and style images are loaded and processed for input into the neural network.
4) Training Setup:
An optimizer (Adam) is used for training the neural network. The train_step function is defined to perform a single step of training, computing the loss and applying gradients.
5) Training Loop:
The model is trained for a specified number of epochs, and the progress is displayed. The output image is displayed at the end of each epoch.
6) Saving and Displaying the Result:
The final stylized image is saved and displayed.

Usage

1) Install the required libraries:
2) Place your content and style images in the working directory and set their paths in the code:
3) Run the script:
python your_script_name.py
4)The output stylized image will be saved as stylized_image.png in the working directory.

References :
1) TensorFlow
2) NumPy
3) Pillow (PIL Fork)
4) Matplotlib
