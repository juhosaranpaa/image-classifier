# Import necessary libraries
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import os
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Load CIFAR-10 dataset: 60,000 32x32 color images in 10 classes
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to the range [0, 1] for better model performance
train_images, test_images = train_images / 255.0, test_images / 255.0

# Define class names for CIFAR-10 categories
class_names = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

# Define the path where the model will be saved or loaded from
model_path = "cifar10_cnn_model.h5"

# Load the model if it already exists, otherwise create a new one
if os.path.exists(model_path):
    model = tf.keras.models.load_model(model_path)
    print("Model loaded from disk.")
else:
    # Define the CNN architecture
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 3)),  # First conv layer
        layers.MaxPooling2D((2, 2)),                                            # First pooling layer
        layers.Conv2D(32, (3, 3), activation="relu"),                           # Second conv layer
        layers.MaxPooling2D((2, 2)),                                            # Second pooling layer
        layers.Flatten(),                                                      # Flatten output for dense layers
        layers.Dense(64, activation="relu"),                                   # Fully connected layer
        layers.Dense(10),                                                      # Output layer for 10 classes
    ])

# Compile the model with Adam optimizer and sparse categorical cross-entropy loss
model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

# Train the model for 10 epochs using training data; validate with test data
model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

# Save the trained model to disk
model.save(model_path)
print("Model saved")

# Function to classify a single image and return the predicted class name
def classify_image(image):
    img_array = tf.expand_dims(image, 0)  # Add batch dimension
    predictions = model.predict(img_array)  # Run prediction
    predicted_class = tf.argmax(predictions[0]).numpy()  # Get index of max probability
    return class_names[predicted_class]  # Return corresponding class name

# Function to display an image with its predicted and true label
def show_image_with_prediction(image, true_label):
    predicted_label = classify_image(image)  # Get model prediction
    plt.figure()  # Create a new figure
    plt.imshow(image)  # Show image
    plt.title(f"Predicted: {predicted_label}, True: {true_label}")  # Add title with labels
    plt.axis("off")  # Hide axis
    plt.show()  # Display the figure

# Select a test image index to visualize
index = 5
image = test_images[index]
true_label = class_names[int(test_labels[index])]

# Show the selected image with prediction
show_image_with_prediction(image, true_label)

# Evaluate the trained model on test data and print accuracy and loss
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f"\nTest Accuracy: {test_acc:.4f}")
print(f"Test Loss: {test_loss:.4f}")

# Predict labels for all test images
y_pred_logits = model.predict(test_images)                # Get logits for each test image
y_pred = np.argmax(y_pred_logits, axis=1)                 # Convert logits to predicted class indices
y_true = test_labels.flatten()                            # Flatten labels to match shape

# Print classification report with precision, recall, f1-score per class
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# Print the confusion matrix to analyze class-wise prediction distribution
print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))
