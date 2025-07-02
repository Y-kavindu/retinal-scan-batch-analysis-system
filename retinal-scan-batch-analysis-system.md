### Project: High-Throughput System for Batch Analysis of Retinal Scans

Here is a comprehensive Google Colab notebook structure designed to address your project requirements. This notebook will guide you through training a deep learning model to classify retinal images and then using that model to perform batch analysis on a directory of new images.

The "Ocular Disease Recognition" (ODIR) dataset from Kaggle is a suitable choice for this task. It contains 5,000 patient cases with fundus photographs of both left and right eyes, annotated with eight different labels, including Normal, Diabetes, and Glaucoma.

**Note on the ODIR Dataset:** This dataset presents some challenges. It is highly imbalanced, with a large number of "Normal" images compared to specific diseases. Additionally, the primary labels correspond to the patient, not necessarily individual eye images, which can lead to complexities in training. For the purpose of this notebook, we will focus on a simplified three-class problem: Normal, Diabetic Retinopathy, and Glaucoma.

---

### Google Colab Notebook

#### 1. Setup

This section installs necessary libraries and imports the required modules for the project.

```python
!pip install tensorflow pandas numpy scikit-learn matplotlib seaborn opendatasets

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import opendatasets as od
```

#### 2. Data Ingestion & Preprocessing

Here, we download the "Ocular Disease Recognition" dataset from Kaggle and preprocess it for our model.

```python
# Download the dataset from Kaggle
od.download("https://www.kaggle.com/andrewmvd/ocular-disease-recognition-odir5k")

# Path to the dataset
data_dir = './ocular-disease-recognition-odir5k/ODIR-5K/ODIR-5K/Training Images'
labels_path = './ocular-disease-recognition-odir5k/ODIR-5K/data.xlsx'

# Load the labels
labels_df = pd.read_excel(labels_path)

# Preprocess labels for our three classes
def preprocess_labels(row):
    if 'normal' in row['Left-Diagnostic Keywords'] or 'normal' in row['Right-Diagnostic Keywords']:
        return 'Normal'
    if 'diabetic retinopathy' in row['Left-Diagnostic Keywords'] or 'diabetic retinopathy' in row['Right-Diagnostic Keywords']:
        return 'Diabetic Retinopathy'
    if 'glaucoma' in row['Left-Diagnostic Keywords'] or 'glaucoma' in row['Right-Diagnostic Keywords']:
        return 'Glaucoma'
    return 'Other'

labels_df['label'] = labels_df.apply(preprocess_labels, axis=1)
labels_df = labels_df[labels_df['label'].isin(['Normal', 'Diabetic Retinopathy', 'Glaucoma'])]

# Create a new dataframe with image filenames and their corresponding labels
image_data = []
for index, row in labels_df.iterrows():
    image_data.append({'filename': row['Left-Fundus'], 'label': row['label']})
    image_data.append({'filename': row['Right-Fundus'], 'label': row['label']})

image_df = pd.DataFrame(image_data)

# Data augmentation and splitting
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2, # 80% training, 20% validation
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

train_generator = train_datagen.flow_from_dataframe(
    dataframe=image_df,
    directory=data_dir,
    x_col="filename",
    y_col="label",
    subset="training",
    batch_size=32,
    seed=42,
    shuffle=True,
    class_mode="categorical",
    target_size=(224,224))

validation_generator = train_datagen.flow_from_dataframe(
    dataframe=image_df,
    directory=data_dir,
    x_col="filename",
    y_col="label",
    subset="validation",
    batch_size=32,
    seed=42,
    shuffle=True,
    class_mode="categorical",
    target_size=(224,224))
```

#### 3. Model Architecture

We will use a pre-trained ResNet50 model and add a custom classification head.

```python
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the base model layers
for layer in base_model.layers:
    layer.trainable = False

# Add custom layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(3, activation='softmax')(x) # 3 classes

model = Model(inputs=base_model.input, outputs=predictions)
```

#### 4. Model Training

This section compiles and trains the model.

```python
model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks
checkpoint = ModelCheckpoint("retinal_model.h5", save_best_only=True, monitor='val_accuracy', mode='max')
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)

history = model.fit(
    train_generator,
    epochs=50,
    validation_data=validation_generator,
    callbacks=[checkpoint, early_stopping, reduce_lr]
)
```

#### 5. Performance Evaluation

Evaluate the model's performance on the validation set.

```python
# Plot training history
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()

# Confusion Matrix and Classification Report
Y_pred = model.predict(validation_generator)
y_pred = np.argmax(Y_pred, axis=1)

print('Confusion Matrix')
cm = confusion_matrix(validation_generator.classes, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.show()

print('Classification Report')
target_names = list(train_generator.class_indices.keys())
print(classification_report(validation_generator.classes, y_pred, target_names=target_names))

```

#### 6. Batch Processing & Inference

This function takes a directory of new images and generates predictions.

```python
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

def batch_predict(image_dir, model_path):
    model = load_model(model_path)
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    results = []

    for filename in image_files:
        img_path = os.path.join(image_dir, filename)
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        predictions = model.predict(img_array)[0]
        results.append({
            'filename': filename,
            'Normal_prob': predictions[0],
            'Diabetic_Retinopathy_prob': predictions[1],
            'Glaucoma_prob': predictions[2]
        })
    return pd.DataFrame(results)

# Example usage:
# Create a dummy directory with some images for demonstration
# !mkdir new_images
# # (add some images to this directory)
# batch_results = batch_predict('new_images', 'retinal_model.h5')
# print(batch_results)
```

#### 7. Results Aggregation & Reporting

This final section provides summary statistics and visualizations of the batch predictions.

```python
def analyze_batch_results(df):
    df['predicted_class'] = df[['Normal_prob', 'Diabetic_Retinopathy_prob', 'Glaucoma_prob']].idxmax(axis=1)
    df['predicted_class'] = df['predicted_class'].str.replace('_prob', '')

    # Summary Statistics
    summary_stats = df['predicted_class'].value_counts(normalize=True) * 100
    print("Summary Statistics (%):")
    print(summary_stats)

    # Visualization
    summary_stats.plot(kind='bar', title='Distribution of Predicted Conditions')
    plt.ylabel('Percentage')
    plt.show()

    # Prioritize High-Risk Cases
    high_risk = df[(df['predicted_class'] == 'Diabetic Retinopathy') | (df['predicted_class'] == 'Glaucoma')]
    high_risk['max_prob'] = high_risk[['Diabetic_Retinopathy_prob', 'Glaucoma_prob']].max(axis=1)
    high_risk_sorted = high_risk.sort_values(by='max_prob', ascending=False)
    print("\nHigh-Risk Cases (sorted):")
    print(high_risk_sorted[['filename', 'predicted_class', 'max_prob']])

# Example usage with the results from the previous step:
# analyze_batch_results(batch_results)
```