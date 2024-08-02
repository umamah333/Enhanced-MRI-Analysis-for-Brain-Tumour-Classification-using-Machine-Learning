# Enhanced-MRI-Analysis-for-Brain-Tumour-Classification-using-Machine-Learning
This repository hosts the code and resources for a project focused on MRI analysis for the classification of brain tumours using machine learning techniques. Leveraging a dataset of MRI images of brain tumors, this project aims to develop and implement advanced algorithms to accurately classify different types of brain tumours. our goal is to create a robust classification model capable of accurately identifying different types of brain tumors based on image features extracted from MRI scans.

# Key Features
Data Preprocessing: Scripts and tools for preprocessing MRI images, including normalization, resizing, and noise reduction.

Feature Extraction: Implementation of feature extraction algorithms to extract relevant features from MRI images, such as texture analysis, intensity histograms, and shape descriptors.

Machine Learning Models: Development and evaluation of machine learning models for brain tumor classification, including traditional classifiers like Support Vector Machines (SVM), Random Forest, and advanced deep learning architectures such as Convolutional Neural Networks (CNN).

# Parameters Used in the code:
several hyperparameters are used both in the CNN (Convolutional Neural Network) and the SVM (Support Vector Machine). Here is a detailed list of the hyperparameters and their values:

Hyperparameters for CNN
Input Shape:

Value: (256, 256, 3)
Description: The shape of the input images. This includes the height and width (256x256 pixels) and the number of channels (3 for RGB).
Conv2D Layers:

Filters: [32, 64, 128]
Description: The number of filters in each convolutional layer. The first layer has 32 filters, the second has 64, and the third has 128.
Kernel Size: (3, 3)
Description: The size of the convolutional kernel.
Activation Function: 'relu'
Description: The activation function used in each convolutional layer.
MaxPooling2D Layers:

Pool Size: (2, 2)
Description: The size of the pooling window.
Flatten Layer:

Description: Flattens the 3D output of the convolutional layers to 1D.
Dense Layers:

Units: 128
Description: The number of neurons in the dense layer.
Activation Function: 'relu'
Description: The activation function used in the dense layer.
Dropout Layer:

Rate: 0.5
Description: The dropout rate used to prevent overfitting.
Output Layer:

Units: 4
Description: The number of classes in the output.
Activation Function: 'softmax'
Description: The activation function used in the output layer.
Compilation Parameters:

Optimizer: 'adam'
Description: The optimizer used to compile the model.
Loss Function: 'categorical_crossentropy'
Description: The loss function used for multi-class classification.
Metrics: ['accuracy']
Description: The metric used to evaluate the model.
Training Parameters:

Epochs: 10 (initially; suggested to increase for better performance)
Description: The number of times the entire dataset is passed through the model.
Batch Size: 32
Description: The number of samples per gradient update.
Hyperparameters for SVM
Kernel:

Value: 'rbf' (Radial Basis Function)
Description: The kernel type used in the SVM model. The RBF kernel is chosen for its ability to handle non-linear data.
StandardScaler:

Description: Standardizes features by removing the mean and scaling to unit variance. This is part of the SVM pipeline to normalize the features.
Data Preprocessing
ImageDataGenerator:

Rescale: 1./255
Description: Rescales the pixel values to the range [0, 1].
Target Size:

Value: (256, 256)
Description: The size to which all input images are resized.
Class Mode:

Value: 'categorical'
Description: The type of label arrays that are returned: one-hot encoded.
Additional Notes
Early Stopping (suggested):
Monitor: 'val_loss'
Patience: 3
Restore Best Weights: True
Description: Stops training when the validation loss stops improving for 3 consecutive epochs and restores the best weights.
These hyperparameters are crucial as they define the architecture, training process, and evaluation metrics for both the CNN and SVM models in your project. Adjusting these hyperparameters can significantly impact the performance and accuracy of your models.

Evaluation Metrics: Calculation and visualization of evaluation metrics like accuracy, precision, recall, and F1-score to assess the performance of classification models.
# Usage
Download the dataset into your google drive and the fetch the dataset, perform pre-process and EDA and then perform the remaining code to analyse the peformance of the model.

# Contributing
Contributions to the project are welcome! If you have any ideas for improvements, bug fixes, or new features, feel free to open an issue or submit a pull request. Please refer to the contribution guidelines for more information.

# Dataset link:
https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset?select=Training

# Acknowledgments
Special thanks to Masoud Nickparvar for providing the Brain Tumor MRI Dataset used in this project.
