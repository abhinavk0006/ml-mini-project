# Song Popularity Prediction Using Machine Learning

A machine-learning project inspired by the research paper **“Predicting Hit Songs Using Repeated Chorus.”** The project investigates whether audio characteristics from a 15-second repeated section can help classify songs as more popular or less popular.

## Project workflow

1. Download the FMA Small audio collection and FMA metadata.
2. Select the 500 tracks with the highest recorded listen counts and the 500 with the lowest recorded listen counts, among tracks whose audio files are available.
3. Extract a representative 15-second section using mel-spectrogram similarity.
4. Extract 38 audio features with Librosa, including MFCC statistics, chroma, RMS energy, spectral centroid, spectral bandwidth, spectral rolloff and zero-crossing rate.
5. Train and evaluate Logistic Regression, SVM (RBF kernel), Random Forest and Gradient Boosting.
6. Compare accuracy, precision, recall and F1-score, and inspect confusion matrices.
7. Run the two added experiments: GridSearchCV tuning for Logistic Regression and Random Forest using five-fold cross-validation.

## Repository contents

- `ML_mini_final.ipynb` — the final project notebook.
- `prepare_dataset.py` — recreates and exports the 1,000-track label/metadata subset after the FMA files have been downloaded.
- `requirements.txt` — Python dependencies.
- `data/README.md` — explains the dataset files and how to generate them.

## Dataset

The project uses **FMA Small** and its metadata. The original audio collection is not included in this repository because it is a separate, large download. Download the FMA Small audio and metadata from the [Free Music Archive dataset page](https://github.com/mdeff/fma).

Extract the archives so the project directory contains:

```text
fma_small/
fma_metadata/tracks.csv
```

Run the dataset preparation script to create `data/song_labels.csv`. This CSV contains the selected track IDs, titles, listen counts, genres, audio paths and binary labels. The label is a project-specific proxy: top 500 recorded listens = `1`, bottom 500 = `0`; it is **not** a Billboard chart label.

The notebook computes the audio feature table when run. It also saves `model_results.csv` in the working directory. Do not commit the full FMA audio collection to GitHub; follow the dataset's terms and attribution requirements.

## Setup

Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
```

Activate the environment, then install dependencies:

```bash
pip install -r requirements.txt
```

Launch Jupyter:

```bash
jupyter notebook ML_mini_final.ipynb
```

Run the notebook cells in order. The first two cells download and extract FMA archives in the current working directory, so ensure sufficient disk space and an internet connection. The repeated-section extraction compares many audio windows and can take a considerable amount of time.

## Algorithms

- **Logistic Regression:** linear baseline classifier.
- **SVM with RBF kernel:** nonlinear classifier; inputs are standardized.
- **Random Forest:** tree ensemble that captures nonlinear relationships without feature scaling.
- **Gradient Boosting:** sequential tree ensemble.

## Added features

1. **Logistic Regression tuning:** GridSearchCV searches regularization strength (`C`) and solver using five-fold cross-validation, optimizing F1-score.
2. **Random Forest tuning:** GridSearchCV searches the number of trees, maximum depth, minimum split size and minimum leaf size, also using five-fold cross-validation and F1-score.

## Evaluation

The notebook uses an 80/20 stratified train-test split and reports accuracy, precision, recall and F1-score. Results may vary with dependency versions or differences in the downloaded metadata/audio files. Use the notebook's actual output values when reporting results.

## Reference

Liu, E. X. et al., *Predicting Hit Songs Using Repeated Chorus*, CS229 Milestone Report, 2021. This project adapts the paper's general idea; its dataset, popularity labels, feature set and implemented algorithms differ from the original study.
