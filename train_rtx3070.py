#!/usr/bin/env python3
"""
RTX 3070 Optimized Training Script
Maximizes GPU performance for fastest, most accurate training
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Add src to path
sys.path.append('src')
from data_collector import DataCollector
from utils.config import *

def setup_rtx3070():
    """Configure TensorFlow for RTX 3070 optimal performance"""
    print("🚀 Setting up RTX 3070 optimization...")
    
    # Check GPU availability
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Enable memory growth to prevent VRAM allocation issues
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # Enable mixed precision for RTX 3070 (Ampere architecture)
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            
            print(f"✅ RTX 3070 GPU acceleration enabled!")
            print(f"✅ Mixed precision enabled for maximum speed!")
            print(f"✅ GPU memory growth enabled!")
            
            return True
            
        except RuntimeError as e:
            print(f"GPU setup error: {e}")
            return False
    else:
        print("❌ No GPU detected! Make sure you're running on the RTX 3070 PC")
        return False

def create_enhanced_model():
    """Create model optimized for gesture distinction"""
    model = keras.Sequential([
        # Input layer
        layers.Input(shape=(63,)),
        
        # Enhanced preprocessing
        layers.LayerNormalization(),
        layers.GaussianNoise(0.01),  # Slight noise for robustness
        
        # Feature extraction layers
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        
        # Output layer (8 classes)
        layers.Dense(8, activation='softmax', dtype='float32')  # Force float32 for mixed precision
    ])
    
    # Advanced optimizer settings for RTX 3070
    optimizer = keras.optimizers.AdamW(
        learning_rate=0.001,
        weight_decay=1e-4,
        beta_1=0.9,
        beta_2=0.999
    )
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def augment_gesture_data(X, y):
    """Advanced data augmentation for gesture distinction"""
    print("🔄 Applying advanced data augmentation...")
    
    augmented_X = []
    augmented_y = []
    
    for i in range(len(X)):
        landmarks = X[i].reshape(21, 3)
        
        # Original sample
        augmented_X.append(X[i])
        augmented_y.append(y[i])
        
        # Augmentation 1: Slight noise (hand tremor simulation)
        noise = np.random.normal(0, 0.008, landmarks.shape)
        noisy = landmarks + noise
        augmented_X.append(noisy.flatten())
        augmented_y.append(y[i])
        
        # Augmentation 2: Scale variation (distance changes)
        scale = np.random.uniform(0.92, 1.08)
        wrist = landmarks[0]
        scaled = wrist + (landmarks - wrist) * scale
        augmented_X.append(scaled.flatten())
        augmented_y.append(y[i])
        
        # Augmentation 3: Rotation (slight hand orientation changes)
        angle = np.random.uniform(-8, 8) * np.pi / 180
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        
        wrist_2d = landmarks[0, :2]
        rotated = landmarks.copy()
        for j in range(21):
            centered = landmarks[j, :2] - wrist_2d
            rotated_coords = rotation_matrix @ centered
            rotated[j, :2] = rotated_coords + wrist_2d
        
        augmented_X.append(rotated.flatten())
        augmented_y.append(y[i])
    
    print(f"✅ Data augmented: {len(X)} → {len(augmented_X)} samples")
    return np.array(augmented_X), np.array(augmented_y)

def train_optimized_model():
    """Train model with RTX 3070 optimization"""
    start_time = time.time()
    
    # Setup GPU
    if not setup_rtx3070():
        print("❌ GPU setup failed. Exiting...")
        return
    
    # Load data
    print("📊 Loading gesture data...")
    collector = DataCollector()
    df = collector.load_existing_data()
    
    if df.empty:
        print("❌ No training data found!")
        print("💡 Run data collection first: python main.py --collect")
        return
    
    # Show data statistics
    print("\n📈 Data Statistics:")
    for gesture in GESTURE_CLASSES:
        count = len(df[df['gesture'] == gesture])
        print(f"  {gesture}: {count} samples")
    
    # Prepare data
    feature_columns = [col for col in df.columns if col.startswith('landmark_')]
    X = df[feature_columns].values
    y = df['gesture'].values
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Apply data augmentation
    X_aug, y_aug = augment_gesture_data(X, y_encoded)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_aug, y_aug, test_size=0.2, random_state=42, stratify=y_aug
    )
    
    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"✅ Training set: {len(X_train_scaled)} samples")
    print(f"✅ Test set: {len(X_test_scaled)} samples")
    
    # Create model
    print("\n🧠 Creating enhanced model...")
    model = create_enhanced_model()
    model.summary()
    
    # Setup callbacks for RTX 3070
    callbacks_list = [
        callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=20,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=8,
            min_lr=1e-7,
            verbose=1
        ),
        callbacks.ModelCheckpoint(
            'models/gesture_classifier_rtx3070.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Train model (RTX 3070 will be FAST!)
    print("\n🚀 Starting training on RTX 3070...")
    training_start = time.time()
    
    history = model.fit(
        X_train_scaled, y_train,
        batch_size=128,  # Large batch size for RTX 3070
        epochs=150,
        validation_data=(X_test_scaled, y_test),
        callbacks=callbacks_list,
        verbose=1
    )
    
    training_time = time.time() - training_start
    print(f"\n⚡ Training completed in {training_time:.1f} seconds!")
    
    # Evaluate model
    print("\n📊 Evaluating model...")
    test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
    print(f"🎯 Test Accuracy: {test_accuracy:.4f}")
    
    # Detailed evaluation
    y_pred = model.predict(X_test_scaled)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred_classes, target_names=label_encoder.classes_))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.title('Confusion Matrix - RTX 3070 Trained Model')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig('models/confusion_matrix_rtx3070.png', dpi=300)
    plt.show()
    
    # Save scaler and label encoder
    import pickle
    with open('models/scaler_rtx3070.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open('models/label_encoder_rtx3070.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    
    total_time = time.time() - start_time
    print(f"\n🎉 Complete training pipeline finished in {total_time:.1f} seconds!")
    print(f"🔥 RTX 3070 Performance: {training_time:.1f}s training time")
    print(f"📈 Final Accuracy: {test_accuracy:.4f}")
    
    # Gesture-specific analysis
    print("\n🖐️ Gesture-Specific Performance:")
    for i, gesture in enumerate(label_encoder.classes_):
        gesture_mask = y_test == i
        if np.any(gesture_mask):
            gesture_accuracy = np.mean(y_pred_classes[gesture_mask] == y_test[gesture_mask])
            print(f"  {gesture}: {gesture_accuracy:.4f}")

def main():
    """Main training function"""
    print("🎯 RTX 3070 Optimized Gesture Training")
    print("="*50)
    
    # Check if we're on the right system
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if not gpus:
        print("⚠️ Warning: No GPU detected!")
        print("💡 Make sure you're running this on your RTX 3070 PC")
        choice = input("Continue anyway? (y/n): ")
        if choice.lower() != 'y':
            return
    
    try:
        train_optimized_model()
        print("\n✅ Training successful!")
        print("💾 Models saved to models/ directory")
        print("🚀 Your RTX 3070 optimized model is ready!")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 