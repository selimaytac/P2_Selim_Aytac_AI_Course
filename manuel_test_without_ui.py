# model_testing.py

import os
import random
import numpy as np
import soundfile as sf
import librosa
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report
from tensorflow.keras.utils import to_categorical

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

def select_random_test_files(data_folder, num_files=300):
    test_files = []
    true_labels = []
    all_labels = set()

    for root, dirs, files in os.walk(data_folder):
        for file in files:
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)
                label = os.path.basename(root)
                test_files.append(file_path)
                true_labels.append(label)
                all_labels.add(label)
    
    combined = list(zip(test_files, true_labels))
    random.shuffle(combined)
    selected_files, selected_labels = zip(*combined[:num_files])
    
    return list(selected_files), list(selected_labels), list(all_labels)

data_folder = 'Sound Source'

test_files, true_labels, all_labels = select_random_test_files(data_folder, num_files=300)

print(f"Selected {len(test_files)} files and {len(true_labels)} labels.")
print("Labels:", true_labels)

X_test_new = []
filtered_labels = []

for file, label in zip(test_files, true_labels):
    features = extract_features(file)
    if features is not None:
        X_test_new.append(features)
        filtered_labels.append(label)

X_test_new = np.array(X_test_new)
if X_test_new.ndim == 2:  
    X_test_new = np.expand_dims(X_test_new, axis=2)  

encoder = LabelEncoder()
encoder.fit(all_labels) 
filtered_labels_encoded = encoder.transform(filtered_labels)
filtered_labels_encoded = to_categorical(filtered_labels_encoded)

model = load_model('best_model.h5')

predictions = model.predict(X_test_new)
predicted_labels = np.argmax(predictions, axis=1)

predicted_labels_decoded = encoder.inverse_transform(predicted_labels)
filtered_labels_decoded = encoder.inverse_transform(np.argmax(filtered_labels_encoded, axis=1))

print(classification_report(filtered_labels_decoded, predicted_labels_decoded))

accuracy = np.sum(predicted_labels == np.argmax(filtered_labels_encoded, axis=1)) / len(filtered_labels)
print(f'Test Accuracy: {accuracy:.4f}')
