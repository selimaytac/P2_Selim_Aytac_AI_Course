import os

data_path = 'Sound Source'

for root, dirs, files in os.walk(data_path):
    print(f'Klasör: {root}')
    for file in files:
        if file.endswith('.wav'):
            print(f'Ses dosyası: {file}')
