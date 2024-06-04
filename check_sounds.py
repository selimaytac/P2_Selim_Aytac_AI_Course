import os

def check_directory(data_folder):
    directory_count = 0
    file_count = 0
    for root, dirs, files in os.walk(data_folder):
        print(f"Checking directory: {root}")
        for dir_name in dirs:
            print(f"Directory: {dir_name}")
            directory_count += 1
        for file_name in files:
            print(f"File: {file_name}")
            file_count += 1
            
    print(f"Total directories: {directory_count}")
    print(f"Total files: {file_count}")
            
data_folder = 'Sound Source'

check_directory(data_folder)
