# CNN Multi-Class Image Classification

## Overview
This project builds a **Convolutional Neural Network (CNN)** using TensorFlow/Keras to perform multi-class image classification. The model classifies images into three categories: `driving_license`, `social_security`, and `others`. It utilizes a combination of convolutional layers, batch normalization, max pooling, and dropout to achieve high performance in classifying these images.

## Aim
The goal of this project is to develop a sequential CNN model that can accurately classify images into one of three predefined classes. The model is built using TensorFlow/Keras and trained on a dataset of images from the categories `driving_license`, `social_security`, and `others`.

## Data
The dataset consists of images categorized into three folders:
- `driving_license`
- `social_security`
- `others`

Each image is resized to `224x224` pixels, and data augmentation techniques such as rotation, zoom, and flips were applied to improve generalization and reduce overfitting.

## CNN Model Structure

The CNN model follows the architecture outlined below:

```python
# Build the CNN model
model= Sequential()

# Input layer
model.add(Input(shape= (224,224,3)))

# First convolutional layer
model.add(Conv2D(32, 3, padding= 'same', activation= 'relu'))

# Batch normalization to normalize the inputs to the next layer
model.add(BatchNormalization())

# Max pooling layer to reduce spacial dimensions of the feature map
model.add(MaxPool2D())

# Dropout layer to prevent overfitting
model.add(Dropout(0.4))

# Flatten the feature map into a 1D vector for the fully connected layer
model.add(Flatten())

# Fully connected layer
model.add(Dense(128, activation= 'relu'))

# Output layer with 3 units (one for each class) and softmax activation for multi-class classification
model.add(Dense(3, activation= 'softmax'))

# Compile the model
model.compile(optimizer= Adam(learning_rate= 0.00001), loss= tf.keras.losses.SparseCategoricalCrossentropy(from_logits= True), metrics= ['accuracy'])
```

### Model Components:
- **Convolutional Layers**: Extract features from the input images using 3x3 kernels.
- **Batch Normalization**: Normalizes the output of the convolutional layers, speeding up training and improving model stability.
- **Max Pooling**: Reduces the spatial dimensions of the feature maps, helping to make the model more efficient.
- **Dropout**: A dropout rate of 40% is used to prevent overfitting by randomly disabling some neurons during training.
- **Fully Connected Layer**: After flattening the feature maps, a dense layer with 128 neurons is applied.
- **Output Layer**: A softmax layer with 3 units for multi-class classification, producing probabilities for each class.

### Results
The model achieved an **accuracy of 86%** on the validation set, with the following detailed performance per class:

| Class             | Precision | Recall | F1-Score | Support |
|-------------------|-----------|--------|----------|---------|
| driving_license    | 0.89      | 0.85   | 0.87     | 40      |
| social_security    | 0.76      | 0.95   | 0.84     | 40      |
| others             | 0.97      | 0.78   | 0.86     | 40      |
| **Overall Accuracy**|           |        | **0.86** | **120** |

### Key Observations:
- **Social Security**: The model achieved high recall (0.95), meaning it correctly identified most `social_security` cases, though precision was slightly lower (0.76), suggesting some false positives.
- **Others**: The model showed strong precision (0.97) but slightly lower recall (0.78), indicating it missed a few true positives for this class.
- **Driving License**: A balanced performance with precision and recall around 0.89 and 0.85, respectively.

### Conclusion
This CNN model provides a robust approach to multi-class image classification with strong overall performance. With further fine-tuning, such as enhancing precision for certain classes, this model could be adapted for broader applications. Advanced techniques such as Transfer Learning or additional data augmentation could further improve model generalization.
