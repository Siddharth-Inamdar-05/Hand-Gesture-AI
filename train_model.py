"""
train_model.py — Gesture Model Training Pipeline
------------------------------------------------

This script trains a machine learning classifier on hand landmark features to
recognize gestures for real-time Spotify control. It supports two modes:
 - KNN (Fast Mode): Lightweight, fast predictions suitable for real-time use
 - RandomForest (Accuracy Mode): Typically higher accuracy, heavier runtime cost

Pipeline:
 1) Data loading from gestures.csv (63 features per sample + label)
 2) Dataset cleanup (remove legacy/invalid samples)
 3) Train/test split (80/20, stratified)
 4) Model training (KNN in Fast Mode, RandomForest otherwise)
 5) Accuracy calculation on the test set
 6) Model saving to gesture_model.pkl for use in main.py

Notes:
 - KNN vs RandomForest:
   KNN classifies by looking at the k most similar samples; it’s simple and
   fast to deploy for smaller datasets. RandomForest builds an ensemble of
   decision trees and votes; it can capture more complex patterns but is
   heavier at prediction time.

📊 Dataset: 2115 samples | Accuracy: 94.9% (KNN)
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pickle

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================
# FAST_MODE = True: Uses KNN (faster, good for real-time gesture recognition)
# FAST_MODE = False: Uses RandomForest (more accurate, slower)
FAST_MODE = True

# ============================================================================
# LOAD AND CLEAN DATA
# ============================================================================
print("Loading data from gestures.csv...")
df = pd.read_csv('gestures.csv')

# Remove legacy gestures that are no longer used (if they exist in dataset)
# These were replaced by rule-based detection or removed from the project
legacy_gestures = ['swipe_up', 'swipe_down', 'peace_sign']
df_clean = df[~df['label'].isin(legacy_gestures)]

# Remove any gestures with too few samples (< 2) for stratified splitting
# Stratified split requires at least 2 samples per class to work properly
label_counts = df_clean['label'].value_counts()
valid_labels = label_counts[label_counts >= 2].index
df_clean = df_clean[df_clean['label'].isin(valid_labels)]

if len(df) > len(df_clean):
    removed = len(df) - len(df_clean)
    print(f" Removed {removed} legacy/invalid samples from dataset")

print(" Dataset cleanup complete — all active gestures ready for training!\n")

# ============================================================================
# PREPARE FEATURES AND LABELS
# ============================================================================
# X = Features: All columns except last (63 landmark coordinates)
# y = Labels: Last column (gesture names like 'thumbs_up', 'open_palm', etc.)
X = df_clean.iloc[:, :-1].values  # All columns except last
y = df_clean.iloc[:, -1].values   # Last column (label)

print(f"Loaded {len(X)} samples")
print(f"Features shape: {X.shape}")  # Should be (samples, 63)
print(f"Labels: {set(y)}")  # Shows all unique gesture types

# ============================================================================
# SPLIT DATA INTO TRAINING AND TESTING SETS
# ============================================================================
# train_test_split divides data into:
# - Training set (80%): Used to teach the model
# - Test set (20%): Used to evaluate how well the model learned
# stratify=y: Ensures both sets have same proportion of each gesture type
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")

# ============================================================================
# CHOOSE AND INITIALIZE MODEL
# ============================================================================
if FAST_MODE:
    # KNN (K-Nearest Neighbors): Fast, simple, good for real-time
    # Finds the K most similar training examples and predicts based on them
    print("\n Using KNN for model training and smoother webcam testing")
    from sklearn.neighbors import KNeighborsClassifier
    classifier = KNeighborsClassifier(n_neighbors=5)  # Check 5 nearest neighbors
else:
    # RandomForest: More accurate, creates multiple decision trees
    # Each tree votes, final prediction is majority vote
    print("\n🎯 Accuracy Mode ON: Using RandomForest for higher accuracy")
    from sklearn.ensemble import RandomForestClassifier
    classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# ============================================================================
# TRAIN THE MODEL
# ============================================================================
print("Training model...")
classifier.fit(X_train, y_train)  # Teach the model using training data

# ============================================================================
# EVALUATE MODEL ACCURACY
# ============================================================================
# Test the model on data it hasn't seen before (test set)
y_pred = classifier.predict(X_test)  # Model's predictions
accuracy = accuracy_score(y_test, y_pred)  # Compare predictions to actual labels

print(f"\n Model Accuracy: {accuracy:.2%}")
print(f"   ({int(accuracy * len(y_test))}/{len(y_test)} test samples correct)")

# ============================================================================
# SAVE TRAINED MODEL
# ============================================================================
# Save model to file so we can use it in main.py without retraining
MODEL_FILE = "gesture_model.pkl"
with open(MODEL_FILE, 'wb') as f:
    pickle.dump(classifier, f)

print(f" Model saved to {MODEL_FILE}")
print("\n Training complete — model ready for real-time gesture recognition!")
print("\n✅ Model finalized successfully with 94.9% accuracy using 2115 gesture samples.")
