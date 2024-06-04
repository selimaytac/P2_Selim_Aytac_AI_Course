# model_training.py

import os
import numpy as np
import soundfile as sf
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.models import Model

def extract_features(file_path):
    try:
        audio, sample_rate = sf.read(file_path)
        if len(audio) < sample_rate * 1: 
            return None
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=22050*2)
        mfccs = librosa.feature.mfcc(y=audio, sr=22050*2, n_mfcc=13)
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def load_data(data_folder='Sound Source'):
    audio_data = []
    labels = []

    for root, dirs, files in os.walk(data_folder):
        for file in files:
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)
                emotion = os.path.basename(root)
                feature = extract_features(file_path)
                if feature is not None:
                    labels.append(emotion)
                    audio_data.append(feature)

    return np.array(audio_data), np.array(labels)

data_folder = 'Sound Source'
audio_data, labels = load_data(data_folder)

if len(audio_data) == 0:
    raise ValueError("No audio files found in the specified directory.")

X_train, X_test, y_train, y_test = train_test_split(audio_data, labels, test_size=0.2, random_state=42)

encoder = LabelEncoder()
y_train_encoded = to_categorical(encoder.fit_transform(y_train))
y_test_encoded = to_categorical(encoder.transform(y_test))

inputs = Input(shape=(X_train.shape[1], 1))
x = LSTM(units=128)(inputs)
x = Dropout(0.5)(x)
outputs = Dense(units=y_train_encoded.shape[1], activation='softmax')(x)
model = Model(inputs, outputs)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

X_train = np.expand_dims(X_train, axis=2)
X_test = np.expand_dims(X_test, axis=2)
model.fit(X_train, y_train_encoded, epochs=64, batch_size=32, validation_split=0.2)

model.summary()

model.save('custom_model.h5')
print("Model saved successfully.")