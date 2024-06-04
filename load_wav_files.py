import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os

data_path = 'Sound Source'

audio_files = []
for root, dirs, files in os.walk(data_path):
    for file in files:
        if file.endswith('.wav'):
            audio_files.append(os.path.join(root, file))

print(f'Toplam {len(audio_files)} ses dosyası bulundu.')

num_files_to_load = 5

for i, file_path in enumerate(audio_files[:num_files_to_load]):
    y, sr = librosa.load(file_path)
    
    plt.figure(figsize=(14, 5))
    librosa.display.waveshow(y, sr=sr)
    plt.title(f'Waveform of file {i+1}: {os.path.basename(file_path)}')
    plt.show()

    S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)

    plt.figure(figsize=(14, 5))
    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Mel-spectrogram of file {i+1}: {os.path.basename(file_path)}')
    plt.show()
