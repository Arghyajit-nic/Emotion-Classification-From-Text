# Emotion Classification from Text

This repository contains an end-to-end implementation of a text-based emotion classification model using logistic regression and TF-IDF features.

## Dataset Used

- **Source:** Kaggle Dataset - "Emotion Detection from Text" by `pashupatigupta`
- **File Type:** CSV
- **Features:**
  - `content`: Input text
  - `sentiment`: Emotion label (e.g., anger, joy, fear, sadness, etc.)

## Approach Summary

### 1. Data Loading
- Dataset is downloaded via `kagglehub`.
- CSV file is read into a pandas DataFrame.


### 2. Preprocessing
- TF-IDF vectorization on the `content` column.
- Used `TfidfVectorizer()` with default parameters.

### 3. Model Training
- Logistic Regression with `max_iter=1000`.
- Trained on TF-IDF vectors of the training data.

### 4. Evaluation
- Evaluated using `accuracy_score` and `classification_report`.
- Achieved 34.9% (~35%) accuracy on the test set.

### 5. Prediction Function
A simple function allows emotion prediction on new input text.

### 6. Gradio Demo (Optional)
An interactive Gradio UI is available in the notebook to test predictions.

### 7. Model Persistence
Both the trained model and the vectorizer are saved using `joblib`.

## Dependencies

- Python 
- pandas
- scikit-learn
- matplotlib
- joblib
- kagglehub
- gradio


