# Detecting Parkinson's Disease from Voice Recordings

ML-2026 project, ESILV · Maxence Périnel, Octave Goudon

Dataset: [UCI – Parkinson's Disease Classification](https://archive.ics.uci.edu/dataset/470/parkinson+s+disease+classification) (Sakar et al., 2019).
756 voice recordings from 252 people (188 Parkinson, 64 healthy), 3 recordings per person.

## Progress

| Week | Task | Status |
|---|---|---|
| 1 | Topic, dataset and project specification | ✅ Done |
| 2 | Clean dataset, graphical analysis, feature analysis, baseline | ✅ Done |
| 3 | AutoML, richer models, probability calibration | ⏳ Next |
| 4 | Final evaluation, report, oral defense | ⏳ Planned |

## Week 2: data cleaning, visualisation and feature analysis

Notebook: [`notebooks/data_cleaning.ipynb`](notebooks/data_cleaning.ipynb)

Each step follows the same method: **test** → **result** → **fix** → **check** (the test is run again).

| Test | Result | Fix |
|---|---|---|
| Column names when reading the file | 2 header rows (`Unnamed: 0`, ...) | read with `header=1` |
| Missing values, data types | 0 missing, all numeric | nothing to fix |
| Duplicated rows | 1 duplicate (person 37) | removed |
| Same label / gender for each person | consistent | nothing to fix |
| Standard deviation of each column | 35 columns ≈ 1e-20 (rounding noise) | removed |
| Final `assert` checks | all passed | – |
| Features that barely change (number of different values) | none (smallest: 103 different values) | nothing to fix |
| Features that are copies of each other (correlation > 0.99) | 97 columns are near-copies (e.g. `ddaShimmer` = 3 × `apq3Shimmer`) | removed in the reduced dataset |
| Does this choice depend on the people used? (two halves of the people) | 81 columns removed with both halves, all 97 found in at least one half | repeated on the training people only in `baseline.ipynb` |

The dataset was already in good shape. Two files are saved in `data/processed/`:
- `pd_speech_clean.csv`: 755 recordings × 717 voice features (+ `id`, `class`, `gender`)
- `pd_speech_reduced.csv`: the same without the 97 near-copies, so 620 voice features

Thresholds 0.95 (241 columns) and 0.90 (364 columns) would remove more. They are tested with a model in `baseline.ipynb` (step 8).

**What the graphs show (for next week):**
- **Imbalanced classes**: 75 % Parkinson / 25 % healthy, so we won't use accuracy alone.
- **No single voice measure separates the classes** (jitter, shimmer, noise ratio overlap), so the model must combine features.
- **Many redundant features** (e.g. `rapJitter` and `ddpJitter`: correlation 1.0), so we compare the full and the reduced datasets.
- **The recordings of one person are similar**, so we'll split the data **by person**.

## Week 2: baseline implementation

Notebook: [`notebooks/baseline.ipynb`](notebooks/baseline.ipynb) (same method: test → result → fix → check)

| Question | Result |
|---|---|
| Split | by person (201 train / 51 test people); the test set is locked until the final evaluation |
| Accuracy alone? | No: a model that always says "Parkinson" gets 0.75 accuracy but 0.5 balanced accuracy |
| Does a split by recording leak? | Yes: AUC 0.809 → 0.896 (logistic regression) when recordings of a person are mixed |
| Best simple model | Logistic regression: balanced accuracy 0.751 ± 0.068, AUC 0.809 ± 0.061 (tree 0.699, k-NN 0.692) |
| Probabilities reliable? | No, over-confident (calibration planned for week 3) |
| One prediction per person (mean of 3 recordings) | better: balanced accuracy 0.754, AUC 0.835, MCC 0.498 |
| Remove near-copy features? | 717 → 618 / 469 / 354 voice features: 0.754 / 0.744 / 0.738 vs 0.751, no real loss (std ≈ 0.07) |
| Gender | imbalanced between classes, but removing it changes nothing (0.751 vs 0.753) |
| Benchmark | Sakar et al. 2019: best 0.86 accuracy / 0.59 MCC (leave-one-subject-out, mRMR top 50); ours 0.81 / 0.50 (not the same protocol) |

## Structure

```
ML-parkinson/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/pd_speech_features.csv      ← original file
│   └── processed/
│       ├── pd_speech_clean.csv         ← clean dataset (717 voice features)
│       ├── pd_speech_reduced.csv       ← without near-copy features (620)
│       └── split_person_ids.csv        ← people in the train / test sets
├── notebooks/
│   ├── data_cleaning.ipynb             ← tests, cleaning, graphs, feature analysis
│   └── baseline.ipynb                  ← split, metrics, baseline models, calibration, benchmark
├── report/                             ← empty for now (final report)
└── ai_governance/
    ├── CONTRIBUTIONS.md                ← empty for now
    ├── PROMPT_LOG.md                   ← started
    └── AI_AUDIT.md                     ← empty for now
```

## How to run

```bash
pip install -r requirements.txt
jupyter notebook notebooks/data_cleaning.ipynb
jupyter notebook notebooks/baseline.ipynb
```
