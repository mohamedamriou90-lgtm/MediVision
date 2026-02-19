"""
MEDIVISION - Training on REAL Medical Images
"""

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import os

print("="*60)
print("🏥 MEDIVISION - TRAINING ON REAL DATA")
print("="*60)

# ============= DATA AUGMENTATION =============
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# ============= LOAD REAL DATA =============
print("\n📊 Loading REAL medical images...")

# Use raw folder
base_path = 'data/raw/chest_xray'

train_generator = train_datagen.flow_from_directory(
    os.path.join(base_path, 'train'),
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    classes=['NORMAL', 'PNEUMONIA']
)

val_generator = val_datagen.flow_from_directory(
    os.path.join(base_path, 'val'),
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    classes=['NORMAL', 'PNEUMONIA']
)

test_generator = test_datagen.flow_from_directory(
    os.path.join(base_path, 'test'),
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    classes=['NORMAL', 'PNEUMONIA'],
    shuffle=False
)

print(f"\n✅ Training samples: {train_generator.samples}")
print(f"✅ Validation samples: {val_generator.samples}")
print(f"✅ Test samples: {test_generator.samples}")

# ============= CREATE MODEL =============
print("\n🧠 Creating neural network...")

# Simple CNN model since ResNet might be too heavy
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2), 
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

model.summary()

# ============= TRAIN =============
print("\n🎯 Training on REAL medical images...")

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        'models/detection/real_model_best.h5',
        monitor='val_accuracy',
        save_best_only=True
    )
]

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=20,
    callbacks=callbacks,
    verbose=1
)

# ============= EVALUATE =============
print("\n📊 Evaluating on test set...")
test_loss, test_acc, test_precision, test_recall = model.evaluate(test_generator)
print(f"✅ Test Accuracy: {test_acc:.2%}")
print(f"✅ Test Precision: {test_precision:.2%}")
print(f"✅ Test Recall: {test_recall:.2%}")

# ============= SAVE =============
model.save('models/detection/medivision_real_final.h5')
print("\n✅ Model saved to models/detection/medivision_real_final.h5")

# ============= PLOT =============
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(history.history['accuracy'], label='Training')
axes[0].plot(history.history['val_accuracy'], label='Validation')
axes[0].set_title(f'Model Accuracy (Final: {test_acc:.2%})')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()

axes[1].plot(history.history['loss'], label='Training')
axes[1].plot(history.history['val_loss'], label='Validation')
axes[1].set_title('Model Loss')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()

plt.tight_layout()
plt.savefig('real_training_history.png')
plt.show()

print("\n" + "="*60)
print("✅✅✅ REAL MODEL TRAINING COMPLETE! ✅✅✅")
print("="*60)