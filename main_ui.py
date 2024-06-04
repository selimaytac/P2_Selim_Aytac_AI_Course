import os
import random
import numpy as np
import soundfile as sf
import librosa
from tkinter import *
from tkinter import ttk, filedialog, messagebox
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
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

def select_random_test_files(data_folder, num_files=50):
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

def start_testing():
    try:
        num_files = int(entry_num_files.get())
        model_path = model_var.get()
        data_folder = entry_data_folder.get()
        
        if not model_path or model_path == "Model bulunamadı":
            messagebox.showwarning("Warning", "Please select a valid model.")
            return
        
        test_files, true_labels, all_labels = select_random_test_files(data_folder, num_files)
        
        print(f"Selected {len(test_files)} files and {len(true_labels)} labels.")
        print("Labels:", true_labels)
        
        X_test_new = []
        filtered_labels = []
        skipped_files = 0

        for file, label in zip(test_files, true_labels):
            features = extract_features(file)
            if features is not None:
                X_test_new.append(features)
                filtered_labels.append(label)
            else:
                skipped_files += 1

        print(f"Processed {len(X_test_new)} files, skipped {skipped_files} files.")
        
        X_test_new = np.array(X_test_new)
        if X_test_new.ndim == 2:  
            X_test_new = np.expand_dims(X_test_new, axis=2) 

        encoder = LabelEncoder()
        encoder.fit(all_labels)
        filtered_labels_encoded = encoder.transform(filtered_labels)
        filtered_labels_encoded = to_categorical(filtered_labels_encoded)

        model = load_model(model_path)

        predictions = model.predict(X_test_new)
        predicted_labels = np.argmax(predictions, axis=1)

        predicted_labels_decoded = encoder.inverse_transform(predicted_labels)
        filtered_labels_decoded = encoder.inverse_transform(np.argmax(filtered_labels_encoded, axis=1))

        detailed_result.delete(1.0, END)
        
        report = classification_report(filtered_labels_decoded, predicted_labels_decoded, output_dict=True)
        
        correct_predictions = 0
        total_predictions = len(filtered_labels_decoded)
        
        for true_label, predicted_label in zip(filtered_labels_decoded, predicted_labels_decoded):
            if true_label == predicted_label:
                detailed_result.insert(END, f'True: {true_label}, Predicted: {predicted_label}\n', 'green')
                correct_predictions += 1
            else:
                detailed_result.insert(END, f'True: {true_label}, Predicted: {predicted_label}\n', 'red')
        
        accuracy = correct_predictions / total_predictions
        detailed_result.insert(END, f'\nTest Accuracy: {accuracy:.4f}')
        
        summary_result.config(state=NORMAL)
        summary_result.delete(1.0, END)
        summary_result.insert(END, f'Total Predictions: {total_predictions}\n')
        summary_result.insert(END, f'Correct Predictions: {correct_predictions}\n')
        summary_result.insert(END, f'Wrong Predictions: {total_predictions - correct_predictions}\n')
        summary_result.insert(END, f'Accuracy: {accuracy:.4f}')
        summary_result.config(state=DISABLED)
    
    except Exception as e:
        messagebox.showerror("Error", str(e))

root = Tk()
root.title("Model Testing")

style = ttk.Style()
style.configure('TButton', font=('Helvetica', 10))
style.configure('TLabel', font=('Helvetica', 10))
style.configure('TEntry', font=('Helvetica', 10))

default_data_folder = 'Sound Source'

Label(root, text="Wav source folder").grid(row=0, column=0, padx=10, pady=10, sticky=E)
entry_data_folder = Entry(root, width=50)
entry_data_folder.insert(0, default_data_folder)
entry_data_folder.grid(row=0, column=1, padx=10, pady=10)

model_files = [f for f in os.listdir('.') if f.endswith('.h5')]
default_model = model_files[0] if model_files else "Model bulunamadı"

Label(root, text="Number of test data to get from folders").grid(row=1, column=0, padx=10, pady=10, sticky=E)
entry_num_files = Entry(root)
entry_num_files.insert(0, "100")
entry_num_files.grid(row=1, column=1, padx=10, pady=10)

Label(root, text="Choose model").grid(row=2, column=0, padx=10, pady=10, sticky=E)
model_var = StringVar(value=default_model)
model_dropdown = ttk.Combobox(root, textvariable=model_var, values=model_files)
model_dropdown.grid(row=2, column=1, padx=10, pady=10)

start_button = ttk.Button(root, text="Start", command=start_testing)
start_button.grid(row=3, column=0, columnspan=2, padx=10, pady=20)

detailed_result_frame = LabelFrame(root, text="Detailed Result", padx=10, pady=10)
detailed_result_frame.grid(row=0, column=2, rowspan=4, padx=10, pady=10, sticky=N+S)

detailed_result = Text(detailed_result_frame, width=70, height=20)
detailed_result.pack()

detailed_result.tag_configure('green', foreground='green')
detailed_result.tag_configure('red', foreground='red')

summary_frame = LabelFrame(root, text="Summary", padx=10, pady=10)
summary_frame.grid(row=4, column=0, columnspan=3, padx=10, pady=10, sticky=E+W)

summary_result = Text(summary_frame, width=80, height=5, state=DISABLED)
summary_result.pack()

root.mainloop()
