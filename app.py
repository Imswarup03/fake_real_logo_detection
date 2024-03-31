import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
import seaborn as sns

# Load the file mapping CSV
df = pd.read_csv('file_mapping.csv')
df.drop('Unnamed: 0', axis=1, inplace=True)

# Define the path to the dataset folder
dataset_folder = 'fakereal-logo-detection-dataset'

# Create empty lists to store images and labels
images = []
labels = []

# Load images and labels from the dataset folder
for index, row in df.iterrows():
    filename = row['Filename']
    label = row['Label']
    img_path = os.path.join(dataset_folder, filename)
    img = cv2.imread(img_path)
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_arr = cv2.resize(img, (224, 224))  # Resize the image to fit MobileNet input size
        image = img_to_array(img_arr)
        image = preprocess_input(image)
        images.append(image)
        labels.append(label)

# Convert lists to numpy arrays
images = np.array(images)
labels = np.array(labels)

# Convert label strings to integers using LabelEncoder
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# Split the dataset into training and testing sets
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)

# Check if the training set is empty
if len(train_images) == 0:
    raise ValueError("The training set is empty. Please check your dataset.")

# Load the MobileNet base model
base_model = tf.keras.applications.MobileNet(include_top=False, weights='imagenet')

# Add custom classification layers on top of MobileNet
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(128, activation='relu')(x)
output = Dense(len(label_encoder.classes_), activation='softmax')(x)  # Use softmax activation for multiclass classification
model = Model(inputs=base_model.input, outputs=output)

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Define early stopping callback
early_stopping = EarlyStopping(monitor="val_loss", patience=5, verbose=1, restore_best_weights=True)

# Train the model
history = model.fit(train_images, train_labels, batch_size=32, epochs=10, 
                    validation_data=(test_images, test_labels), callbacks=[early_stopping])

# Evaluate the model
_, accuracy = model.evaluate(test_images, test_labels)
print('Test Accuracy:', accuracy)

# Generate predictions
y_pred = np.argmax(model.predict(test_images), axis=-1)

# Print classification report
print(classification_report(test_labels, y_pred, target_names=label_encoder.classes_))

# Plot training and validation accuracy over epochs
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()

# Plot training and validation loss over epochs
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# Plot confusion matrix
conf_matrix = confusion_matrix(test_labels, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# Show sample predictions with images
plt.figure(figsize=(12, 8))
for i in range(6):  # Show first 6 sample predictions
    plt.subplot(2, 3, i + 1)
    plt.imshow(test_images[i])
    true_label = label_encoder.inverse_transform([test_labels[i]])[0]
    pred_label = label_encoder.inverse_transform([y_pred[i]])[0]
    plt.title(f'True: {true_label} ({"Fake" if true_label == "0" else "Genuine"}), Predicted: {pred_label} ({"Fake" if pred_label == "0" else "Genuine"})')
    plt.axis('off')
plt.show()

# Get the filenames, true labels, and predicted labels
filenames = df['Filename'].tolist()
true_labels = label_encoder.inverse_transform(test_labels)
predicted_labels = label_encoder.inverse_transform(y_pred)

# Check if the number of filenames matches the number of images
if len(df['Filename']) != len(images):
    raise ValueError("The number of filenames does not match the number of images.")

# Create a DataFrame with image filenames, true labels, and predicted labels
output_df = pd.DataFrame({'Filename': df['Filename'], 'True_Label': true_labels, 'Predicted_Label': predicted_labels})


# # Create a DataFrame with image filenames, true labels, and predicted labels
# output_df = pd.DataFrame({'Filename': filenames, 'True_Label': true_labels, 'Predicted_Label': predicted_labels})

# Save the DataFrame to a CSV file
output_df.to_csv('output.csv', index=False)
