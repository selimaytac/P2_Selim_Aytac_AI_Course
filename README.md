
# P2_Selim_Aytac_AI_Course

This repository contains the code and resources for the AI Course project developed by Selim Aytac. The project includes various components, such as `main_ui`, which are integral to the functionality of the application.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
  - [main_ui](#main_ui)
- [Project Structure](#project-structure)

## Installation

To get started with this project, clone the repository to your local machine:

```bash
git clone https://github.com/selimaytac/P2_Selim_Aytac_AI_Course.git
cd P2_Selim_Aytac_AI_Course
```

Make sure you have the necessary dependencies installed. You can install them using the following command:

```bash
pip install -r requirements.txt
```

## Usage

### main_ui

The `main_ui` module is the main user interface component of the project. It provides the primary interface for interacting with the AI models and visualizing the results.

To use `main_ui`, follow these steps:

1. **Run the main_ui script:**

   ```bash
   python main_ui.py
   ```

2. **Navigate the UI:**
   - The main screen will display various options to load model, test data count, and test models.
   - Use the buttons and input fields to interact with the system. Detailed instructions for each section are provided within the UI.

3. **Load and preprocess data:**
   - Data Source is used to preprocess the data
   - Preprocess the data using the provided tools to clean and prepare it for testing.

4. **Test models:**
   - In the main UI, you can only select the dataset and test the model.
   - Training is done manually via the console. Run the necessary commands in the console to train your models and monitor progress. "python model_builder.py"

5. **Visualize results:**
   - Once the testing is complete, use the visualization tools to analyze the model's performance and results.

## Project Structure

The repository is structured as follows:

- `main_ui.py`: Main user interface script.
- `model_builder.py`: Buildings model with Sound Source Data
- `check_sounds.py`: Checks count of the sound sources
- `load_wav_files.py/`: Loads the WAV files from the Data Source
- `Sound Source`: All Sound sources
- `requirements.txt`: List of dependencies required for the project.
- `manuel_test_without_ui.py`: Test model without UI and console output.
