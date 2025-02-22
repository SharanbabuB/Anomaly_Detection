import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.optimizers import Adam

# Directory paths
train_dirs = [
    r'E:\Anomaly Detection\Dataset\Train',
    r'E:\Anomaly Detection\Dataset\Test'  # Add additional training directory here
]

# Parameters
img_size = (128, 128)
batch_size = 32
epochs = 50

# Define the autoencoder model
input_img = Input(shape=(img_size[0], img_size[1], 3))

# Encoder
x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)

# Decoder
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer=Adam(), loss='mse')

# Training Function
def generate_frames_from_videos(folders, batch_size):
    while True:
        frames = []
        for folder in folders:
            for subfolder in os.listdir(folder):
                subfolder_path = os.path.join(folder, subfolder)
                if os.path.isdir(subfolder_path):
                    for filename in os.listdir(subfolder_path):
                        video_path = os.path.join(subfolder_path, filename)
                        cap = cv2.VideoCapture(video_path)
                        while cap.isOpened():
                            ret, frame = cap.read()
                            if not ret:
                                break
                            frame = cv2.resize(frame, img_size)
                            frame = frame.astype("float32") / 255.0
                            frames.append(frame)
                            if len(frames) == batch_size:
                                yield np.array(frames), np.array(frames)
                                frames = []
                        cap.release()
        if len(frames) > 0:
            yield np.array(frames), np.array(frames)

# Train the autoencoder and capture the training history
history = autoencoder.fit(
    generate_frames_from_videos(train_dirs, batch_size),
    epochs=epochs,
    steps_per_epoch=100,  # Adjust this based on your dataset size
    validation_data=generate_frames_from_videos(train_dirs, batch_size),
    validation_steps=10   # Adjust this based on your dataset size
)

# Save the model
autoencoder.save('anomaly_detection_autoencoder.h5')

# Plot training & validation loss values
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper right')

# Plot "accuracy" as 1 - loss
plt.subplot(1, 2, 2)
plt.plot(1 - np.array(history.history['loss']))
plt.plot(1 - np.array(history.history['val_loss']))
plt.title('Model Accuracy (1 - Loss)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='lower right')

plt.show()

# Generate predictions on test data
def compute_reconstruction_error(autoencoder, test_data):
    reconstructions = autoencoder.predict(test_data)
    # Mean squared error between original and reconstructed images
    return np.mean(np.square(test_data - reconstructions), axis=(1, 2, 3))

# Prepare test data
test_data, _ = next(generate_frames_from_videos([r'E:\Anomaly Detection\Dataset\Test'], batch_size=500))  # Load test frames

# Compute reconstruction errors
reconstruction_errors = compute_reconstruction_error(autoencoder, test_data)

# Assume normal data has low error, anomalous data has high error
threshold = np.percentile(reconstruction_errors, 95)  # Set threshold at 95th percentile

# Create labels: 0 for normal (low error), 1 for anomaly (high error)
labels = np.zeros_like(reconstruction_errors)
labels[reconstruction_errors > threshold] = 1

# Generate ground truth (for demonstration purposes, assuming half are normal and half are anomalies)
ground_truth = np.concatenate((np.zeros(len(labels) // 2), np.ones(len(labels) // 2)))

# Confusion matrix
cm = confusion_matrix(ground_truth, labels)
print("Confusion Matrix:\n", cm)

# Classification report
print("\nClassification Report:\n", classification_report(ground_truth, labels))

# ROC Curve
fpr, tpr, thresholds = roc_curve(ground_truth, reconstruction_errors)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()

# Error distribution
plt.figure()
plt.hist(reconstruction_errors[ground_truth == 0], bins=50, alpha=0.5, label='Normal')
plt.hist(reconstruction_errors[ground_truth == 1], bins=50, alpha=0.5, label='Anomalous')
plt.title('Reconstruction Error Distribution')
plt.xlabel('Reconstruction Error')
plt.ylabel('Frequency')
plt.legend()
plt.show()
