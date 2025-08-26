import os
import pandas as pd
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, GlobalAveragePooling1D
from tensorflow.keras.utils import to_categorical

# --- Step 1: Define a function to extract features from audio ---
# We will extract Mel-Frequency Cepstral Coefficients (MFCCs).
# MFCCs are a feature widely used in automatic speech recognition. 🎵
def extract_features(file_path):
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast') 
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs_processed = np.mean(mfccs.T,axis=0)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None 
    return mfccs_processed

# --- Step 2: Load Data and Extract Features ---
# We will loop through the dataset folder and process each audio file.

DATASET_PATH = 'ravdess_data'
features = []

# RAVDESS emotion labels from filename
emotion_labels = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
    '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
}

print("Starting feature extraction...")
# Iterate through each actor's folder
for actor_folder in os.listdir(DATASET_PATH):
    actor_path = os.path.join(DATASET_PATH, actor_folder)
    if os.path.isdir(actor_path):
        # Iterate through each audio file in the actor's folder
        for file_name in os.listdir(actor_path):
            file_path = os.path.join(actor_path, file_name)
            
            # Extract emotion label from filename
            label = emotion_labels[file_name.split('-')[2]]
            
            # Extract MFCC features
            data = extract_features(file_path)
            if data is not None:
                features.append([data, label])

print("Feature extraction completed.")

if len(features) == 0:
    print("No features extracted. Please check your audio files and dependencies.")
    exit()

# Convert the features list to a Pandas DataFrame
features_df = pd.DataFrame(features, columns=['feature', 'label'])

# --- Step 3: Prepare Data for the Model ---
# We need to format our data so the model can use it. 🧠

# Separate features (X) and labels (y)
X = np.array(features_df['feature'].tolist())
y = np.array(features_df['label'].tolist())

# Encode the categorical text labels to numerical data
le = LabelEncoder()
y_encoded = to_categorical(le.fit_transform(y))

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Reshape X for the 1D CNN model. A channel dimension is needed.
X_train = np.expand_dims(X_train, axis=2)
X_test = np.expand_dims(X_test, axis=2)


# --- Step 4: Build the 1D CNN Model ---
# We use a simple Convolutional Neural Network, which is effective for sequence data like this.
model = Sequential([
    Conv1D(256, 5, padding='same', activation='relu', input_shape=(X_train.shape[1], 1)),
    MaxPooling1D(pool_size=5),
    Dropout(0.3),
    
    Conv1D(128, 5, padding='same', activation='relu'),
    MaxPooling1D(pool_size=5),
    Dropout(0.3),
    
    GlobalAveragePooling1D(),
    
    Dense(128, activation='relu'),
    Dropout(0.3),
    
    Dense(y_encoded.shape[1], activation='softmax') # Output layer
])

# Compile the model
model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

model.summary()

# --- Step 5: Train the Model ---
# This step might take a few minutes depending on your computer. ⏳
print("\nTraining the model...")
history = model.fit(X_train, y_train, 
                    epochs=100, 
                    batch_size=32, 
                    validation_data=(X_test, y_test),
                    verbose=1)

# --- Step 6: Evaluate the Model ---
# Check the model's performance on the test data. 📊
print("\nEvaluating the model...")
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {accuracy*100:.2f}%")
print(f"Test Loss: {loss:.4f}")