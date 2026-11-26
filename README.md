# Detecting Parkinson's Disease from Voice Recordings

ML project, ESILV 2026 · Maxence Périnel, Octave Goudon

We use the [UCI Parkinson's Disease Classification](https://archive.ics.uci.edu/dataset/470/parkinson+s+disease+classification) dataset (Sakar et al., 2019): 756 voice recordings from 252 people (188 with Parkinson, 64 healthy), 3 recordings per person. The goal is to predict from the voice if someone has Parkinson.

## Progress

- **Week 1:** topic, dataset and project specification. Done.
- **Week 2:** data cleaning, data analysis and baseline. Done.
- **Week 3:** AutoML, better models, probability calibration. Next.
- **Week 4:** final evaluation, report, oral defense. Planned.

## Data analysis

Notebook: [`notebooks/data_cleaning.ipynb`](notebooks/data_cleaning.ipynb)

The data was already in good shape. What we found and fixed:

- The file has 2 header rows, so we read it with `header=1`.
- No missing values and all columns are numeric.
- 1 duplicated recording (person 37), removed.
- 35 columns that are only rounding noise (std around 1e-20), removed.
- 97 columns are near-copies of other columns (correlation > 0.99, for example `ddaShimmer` = 3 × `apq3Shimmer`). They are removed in a second, reduced file.

We save two files in `data/processed/`:

- `pd_speech_clean.csv`: 755 recordings, 717 voice features.
- `pd_speech_reduced.csv`: same without the near-copies, 620 voice features.

What the graphs show:

- The classes are imbalanced (75% Parkinson, 25% healthy), so accuracy alone is not enough.
- No single feature separates the two classes, so the model has to combine many of them.
- The 3 recordings of one person look alike, so we have to split the data by person.

## Baseline

Notebook: [`notebooks/baseline.ipynb`](notebooks/baseline.ipynb)

- **Split by person:** 201 people for training, 51 for testing. The test set is not touched until the final evaluation.
- **Why not accuracy:** a model that always says "Parkinson" gets 0.75 accuracy, but only 0.5 balanced accuracy.
- **Why split by person:** if we mix the recordings of a person, the AUC goes from 0.809 to 0.896 (the model recognises the person, not the disease).
- **Best simple model:** logistic regression, balanced accuracy 0.751 ± 0.068 and AUC 0.809 ± 0.061 (decision tree: 0.699, k-NN: 0.692).
- **One prediction per person** (mean of the 3 recordings): balanced accuracy 0.754, AUC 0.835, MCC 0.498.
- **Removing near-copy features** does not change much (0.754 / 0.744 / 0.738 vs 0.751).
- **Gender** is imbalanced between the classes, but removing it changes nothing (0.751 vs 0.753).
- **Probabilities** are over-confident, so we will calibrate them in week 3.
- **Paper benchmark:** Sakar et al. get 0.86 accuracy and 0.59 MCC, we get 0.81 and 0.50. The protocol is not the same, so it is only a rough comparison.

## Structure

```
ML-parkinson/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/pd_speech_features.csv      original file
│   └── processed/
│       ├── pd_speech_clean.csv         clean dataset
│       ├── pd_speech_reduced.csv       without near-copy features
│       └── split_person_ids.csv        people in train / test
├── notebooks/
│   ├── data_cleaning.ipynb             cleaning, graphs, feature analysis
│   └── baseline.ipynb                  split, metrics, baseline models
├── report/                             final report (empty for now)
└── ai_governance/
    ├── CONTRIBUTIONS.md
    ├── PROMPT_LOG.md
    └── AI_AUDIT.md
```

## How to run

```bash
pip install -r requirements.txt
jupyter notebook notebooks/data_cleaning.ipynb
jupyter notebook notebooks/baseline.ipynb
```
