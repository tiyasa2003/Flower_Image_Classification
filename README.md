A deep learning project that classifies flower images into **Daisy, Sunflower, Tulip, Dandelion, and Rose** using a **Convolutional Neural Network (CNN)** with TensorFlow/Keras. Includes preprocessing, data augmentation, model training, evaluation, and visualization of results. Perfect for learning computer vision and deep learning applications.

# Features

* **Data Preparation & Preprocessing:**

  * Load and resize images using OpenCV
  * Normalize pixel values and encode labels
  * Train-test split

* **Data Augmentation:**

  * Rotations, zooms, flips, and shifts using `ImageDataGenerator`

* **Model Architecture:**

  * Multiple Conv2D + MaxPooling layers
  * Dense layers with ReLU and softmax activation

* **Training:**

  * Optimizer: Adam with learning rate scheduler (`ReduceLROnPlateau`)
  * Batch size: 64, Epochs: 10
  * Real-time augmentation applied during training

* **Evaluation & Visualization:**

  * Training/validation accuracy and loss plots
  * Correctly and incorrectly classified samples
  * Confusion matrix for class-wise performance


# Usage

1. Organize your dataset in the following structure:

```
flowers/
├── daisy/
├── sunflower/
├── tulip/
├── dandelion/
└── rose/
```

2. Run the main script:

```bash
python flower_classifier.py
```

3. The model will train and display:

   * Training & validation accuracy/loss
   * Correctly and incorrectly classified images
   * Confusion matrix

# Results

* Accurate multi-class flower classification
* Visualization of model performance
* Ready for further improvements like transfer learning or deployment


# References

* [TensorFlow Keras Documentation](https://www.tensorflow.org/guide/keras)
* [Scikit-learn Documentation](https://scikit-learn.org/stable/)
* [ImageDataGenerator](https://keras.io/api/preprocessing/image/)
