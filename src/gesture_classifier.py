"""
Gesture Classification Module using TensorFlow/Keras
Trains and performs inference on hand gesture data
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import pickle
import os
from typing import Tuple, List, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from data_collector import DataCollector
from utils.config import (
    MODEL_PATH, INPUT_SIZE, HIDDEN_UNITS, DROPOUT_RATE, 
    LEARNING_RATE, EPOCHS, BATCH_SIZE, GESTURE_CLASSES,
    CONFIDENCE_THRESHOLD, MODELS_DIR
)


class GestureClassifier:
    """
    Deep learning model for gesture classification based on hand landmarks
    """
    
    def __init__(self):
        self.model = None
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        self.input_scaler = None
        
        # Setup label encoder with known classes
        self.label_encoder.fit(GESTURE_CLASSES)
        
        # Try to load existing model
        self.load_model()
    
    def create_model(self) -> keras.Model:
        """
        Create the neural network architecture
        
        Returns:
            Compiled Keras model
        """
        model = keras.Sequential([
            # Input layer
            layers.Input(shape=(INPUT_SIZE,)),
            
            # Feature normalization
            layers.LayerNormalization(),
            
            # Hidden layers with dropout for regularization
            layers.Dense(HIDDEN_UNITS[0], activation='relu'),
            layers.Dropout(DROPOUT_RATE),
            layers.BatchNormalization(),
            
            layers.Dense(HIDDEN_UNITS[1], activation='relu'),
            layers.Dropout(DROPOUT_RATE),
            layers.BatchNormalization(),
            
            layers.Dense(HIDDEN_UNITS[2], activation='relu'),
            layers.Dropout(DROPOUT_RATE),
            
            # Output layer
            layers.Dense(len(GESTURE_CLASSES), activation='softmax')
        ])
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def prepare_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load and prepare training data
        
        Returns:
            X_train, X_test, y_train, y_test
        """
        print("Loading gesture data...")
        collector = DataCollector()
        df = collector.load_existing_data()
        
        if df.empty:
            raise ValueError("No training data found! Please collect gesture data first.")
        
        print(f"Loaded {len(df)} samples")
        
        # Separate features and labels
        feature_columns = [col for col in df.columns if col.startswith('landmark_')]
        X = df[feature_columns].values
        y = df['gesture'].values
        
        # Encode labels
        y_encoded = self.label_encoder.transform(y)
        
        # Normalize features
        from sklearn.preprocessing import StandardScaler
        self.input_scaler = StandardScaler()
        X_normalized = self.input_scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_normalized, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        print(f"Training samples: {len(X_train)}")
        print(f"Testing samples: {len(X_test)}")
        print(f"Feature dimensions: {X_train.shape[1]}")
        
        return X_train, X_test, y_train, y_test
    
    def train(self, validation_split: float = 0.2, verbose: int = 1) -> keras.callbacks.History:
        """
        Train the gesture classification model
        
        Args:
            validation_split: Fraction of training data to use for validation
            verbose: Verbosity level for training
            
        Returns:
            Training history
        """
        print("Preparing training data...")
        X_train, X_test, y_train, y_test = self.prepare_data()
        
        print("Creating model...")
        self.model = self.create_model()
        
        print(f"Model architecture:")
        self.model.summary()
        
        # Setup callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=10, restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6
            ),
            keras.callbacks.ModelCheckpoint(
                MODEL_PATH, monitor='val_accuracy', save_best_only=True
            )
        ]
        
        print("Starting training...")
        history = self.model.fit(
            X_train, y_train,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=verbose
        )
        
        # Evaluate on test set
        print("\nEvaluating on test set...")
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"Test accuracy: {test_accuracy:.4f}")
        
        # Generate classification report
        y_pred = self.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        print("\nClassification Report:")
        print(classification_report(
            y_test, y_pred_classes, 
            target_names=self.label_encoder.classes_
        ))
        
        # Save the scaler
        scaler_path = os.path.join(MODELS_DIR, "input_scaler.pkl")
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.input_scaler, f)
        
        self.is_trained = True
        print(f"\nModel saved to: {MODEL_PATH}")
        print(f"Scaler saved to: {scaler_path}")
        
        return history
    
    def predict(self, landmarks: np.ndarray) -> Tuple[str, float]:
        """
        Predict gesture from hand landmarks
        
        Args:
            landmarks: Hand landmark coordinates (63-dimensional array)
            
        Returns:
            Tuple of (predicted_gesture, confidence)
        """
        if not self.is_trained or self.model is None:
            return "none", 0.0
        
        # Ensure landmarks is the right shape
        if landmarks.shape != (INPUT_SIZE,):
            print(f"Warning: Expected {INPUT_SIZE} features, got {landmarks.shape}")
            return "none", 0.0
        
        # Normalize input
        if self.input_scaler is not None:
            landmarks_normalized = self.input_scaler.transform(landmarks.reshape(1, -1))
        else:
            landmarks_normalized = landmarks.reshape(1, -1)
        
        # Predict
        predictions = self.model.predict(landmarks_normalized, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class_idx]
        
        # Apply confidence threshold
        if confidence < CONFIDENCE_THRESHOLD:
            return "none", confidence
        
        predicted_gesture = self.label_encoder.inverse_transform([predicted_class_idx])[0]
        
        return predicted_gesture, confidence
    
    def predict_batch(self, landmarks_batch: np.ndarray) -> List[Tuple[str, float]]:
        """
        Predict gestures for a batch of landmark data
        
        Args:
            landmarks_batch: Batch of landmark arrays
            
        Returns:
            List of (gesture, confidence) tuples
        """
        if not self.is_trained or self.model is None:
            return [("none", 0.0)] * len(landmarks_batch)
        
        # Normalize inputs
        if self.input_scaler is not None:
            landmarks_normalized = self.input_scaler.transform(landmarks_batch)
        else:
            landmarks_normalized = landmarks_batch
        
        # Predict
        predictions = self.model.predict(landmarks_normalized, verbose=0)
        
        results = []
        for pred in predictions:
            predicted_class_idx = np.argmax(pred)
            confidence = pred[predicted_class_idx]
            
            if confidence < CONFIDENCE_THRESHOLD:
                results.append(("none", confidence))
            else:
                predicted_gesture = self.label_encoder.inverse_transform([predicted_class_idx])[0]
                results.append((predicted_gesture, confidence))
        
        return results
    
    def load_model(self) -> bool:
        """
        Load trained model from file
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if os.path.exists(MODEL_PATH):
                self.model = keras.models.load_model(MODEL_PATH)
                
                # Load scaler
                scaler_path = os.path.join(MODELS_DIR, "input_scaler.pkl")
                if os.path.exists(scaler_path):
                    with open(scaler_path, 'rb') as f:
                        self.input_scaler = pickle.load(f)
                
                self.is_trained = True
                print(f"Model loaded from: {MODEL_PATH}")
                return True
        except Exception as e:
            print(f"Failed to load model: {e}")
        
        return False
    
    def save_model(self, path: str = MODEL_PATH):
        """
        Save the trained model
        
        Args:
            path: Path to save the model
        """
        if self.model is not None:
            self.model.save(path)
            print(f"Model saved to: {path}")
        else:
            print("No model to save!")
    
    def plot_training_history(self, history: keras.callbacks.History):
        """
        Plot training history
        
        Args:
            history: Training history from model.fit()
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot accuracy
        ax1.plot(history.history['accuracy'], label='Training Accuracy')
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot loss
        ax2.plot(history.history['loss'], label='Training Loss')
        ax2.plot(history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(MODELS_DIR, 'training_history.png'))
        plt.show()
    
    def evaluate_model(self) -> dict:
        """
        Evaluate model performance on test data
        
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.is_trained:
            return {"error": "Model not trained"}
        
        print("Evaluating model...")
        X_train, X_test, y_train, y_test = self.prepare_data()
        
        # Predictions
        y_pred = self.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        accuracy = accuracy_score(y_test, y_pred_classes)
        precision = precision_score(y_test, y_pred_classes, average='weighted')
        recall = recall_score(y_test, y_pred_classes, average='weighted')
        f1 = f1_score(y_test, y_pred_classes, average='weighted')
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred_classes)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.label_encoder.classes_,
                    yticklabels=self.label_encoder.classes_)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(os.path.join(MODELS_DIR, 'confusion_matrix.png'))
        plt.show()
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm.tolist()
        }


def train_model():
    """Train a new gesture classification model"""
    classifier = GestureClassifier()
    
    try:
        history = classifier.train()
        classifier.plot_training_history(history)
        
        # Evaluate model
        metrics = classifier.evaluate_model()
        print("\nFinal Model Performance:")
        for metric, value in metrics.items():
            if metric != 'confusion_matrix':
                print(f"{metric}: {value:.4f}")
        
    except Exception as e:
        print(f"Training failed: {e}")


def test_classifier():
    """Test the gesture classifier"""
    classifier = GestureClassifier()
    
    if not classifier.is_trained:
        print("No trained model found! Please train the model first.")
        return
    
    print("Testing gesture classifier...")
    print("Available gestures:", GESTURE_CLASSES)
    
    # Test with random data
    for gesture in GESTURE_CLASSES[:5]:  # Test first 5 gestures
        # Generate random landmark data (for testing)
        test_landmarks = np.random.random(INPUT_SIZE)
        
        predicted_gesture, confidence = classifier.predict(test_landmarks)
        print(f"Test input -> Predicted: {predicted_gesture} (confidence: {confidence:.3f})")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "train":
            train_model()
        elif sys.argv[1] == "test":
            test_classifier()
        elif sys.argv[1] == "evaluate":
            classifier = GestureClassifier()
            metrics = classifier.evaluate_model()
            print("Evaluation complete!")
        else:
            print("Usage: python gesture_classifier.py [train|test|evaluate]")
    else:
        print("Gesture Classifier Module")
        print("Use: python gesture_classifier.py train - to train model")
        print("Use: python gesture_classifier.py test - to test model")
        print("Use: python gesture_classifier.py evaluate - to evaluate model") 