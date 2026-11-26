# PROMPT_LOG

Tool used: Claude (Anthropic), through claude.ai and Claude Code.

The prompts below are reformulated by Maxence Périnel for clarity (the original messages were written quickly, partly in French). They keep the same requests. For each entry, what matters is what the AI produced and how we checked it.

## #1 – Repository architecture (2026-11-20 and 2026-11-21)

Prompt:
> Here are the project brief, the governance policy, the project guidelines and our week 1 specification. Create a GitHub repository that follows exactly the archive structure required in the governance policy. Keep the files we do not use yet empty. The goal of this week is a clean dataset, so keep the repository simple.

What the AI produced: the folder structure (`requirements.txt`, `notebooks/`, `report/`, `ai_governance/` with `CONTRIBUTIONS.md`, `PROMPT_LOG.md`, `AI_AUDIT.md`), a `data/` folder and a `README.md`. The unused files (`report/`, `CONTRIBUTIONS.md`, `AI_AUDIT.md`) are empty.

How we checked it: we compared the structure with section 2 of the governance policy.

## #2 – Data cleaning notebook (2026-11-20 to 2026-11-21)

Prompt:
> Write a notebook that cleans the Parkinson dataset. For each step, show the test that finds the problem, the result, the fix, and a check that the problem is gone. Keep the code as simple as possible. Add only graphs that are useful for the analysis, and audit them yourself.

What the AI produced: `notebooks/data_cleaning.ipynb` with 10 steps. The tests found a double header row, one duplicated recording (person 37) and 35 constant columns, plus checks of missing values, data types and label consistency. Four graphs: class balance, boxplots of voice measures, correlation heatmap, and the recordings of single people. The notebook saves `pd_speech_clean.csv`.

How we checked it: `assert` checks at the end of the notebook, each number written in the text comes from a code cell, and we ran the notebook from top to bottom. The AI corrected several of its own drafts (a wrong claim about two jitter features, graphs squashed by outliers, a sentence on recordings that was too strong).

## #3 – Feature analysis (2026-11-21)

Prompt:
> My teacher suggested analysing which features we keep and which we remove, and showing that removing the negligible ones changes nothing. Do this analysis in the same simple style, and also test removing more features than the obvious ones.

What the AI produced: steps 8 and 9 of the cleaning notebook (near-constant features, near-copies with a correlation threshold of 0.99, thresholds 0.95 and 0.90 as more aggressive options, a stability test on two halves of the people) and the file `pd_speech_reduced.csv`.

How we checked it: the AI pointed out that this filter was computed on all people, which is not strictly correct before a train/test split. It is therefore recomputed on the training people only in the baseline notebook (#4).

## #4 – Baseline notebook (2026-11-21 and 2026-11-22)

Prompt:
> Add the baseline implementation in the same style, with explanations. Split the data by person, choose metrics that suit an imbalanced dataset, test that a wrong split inflates the scores, compare a few simple models, check the calibration of the probabilities.

What the AI produced: `notebooks/baseline.ipynb` (10 steps) and `data/processed/split_person_ids.csv`. Person-level split (201 train / 51 test people), the dummy model showing the problem with accuracy, the leakage test, four simple models, calibration, per-person prediction, feature selection on the training people, gender, and comparison with the paper.

How we checked it: cross-validation grouped by person, the scaling placed in a pipeline, all numbers in the text taken from the outputs of the code cells, and a full run of the notebook with no error or warning. The AI removed a hyperparameter curve (tuning belongs to week 3) and the notebook was tested on a Windows computer, where the parallel option (`n_jobs=-1`) failed and was removed.

## #5 – Writing the explanations in French and English (2026-11-26)

Prompt (reformulated by Maxence Périnel, original message in French):
> Write the markdown cells (French and English) of baseline.ipynb and data_cleaning.ipynb: 2 to 4 sentences per cell, simple. Do not change any code or any output, only quote numbers that are in the outputs of the cells, keep the limitations we already wrote, and do not say "better" when the gap is smaller than the standard deviation.

What the AI produced: new French and English texts for all the explanation cells (23 in the cleaning notebook, 20 in the baseline notebook), shorter and without the fixed test / result / fix / check pattern. While rewriting, it removed or softened claims that the outputs do not prove: "the logistic regression is the best model" (the gaps with the tree and the k-NN, 0.05 to 0.06, are smaller than the standard deviation of about 0.07), "removing near-copies costs nothing" (now "we see no clear difference"), "mean of these bands is 0 by construction", "researchers recorded everyone in the same place", "entropy coefficients 8, 9 and 10 are almost identical" (only 8 and 9 were tested), and the person ids 1, 4 and 7 read on a graph. It also noted that the per-person scores have no standard deviation, so the improvement is described with caution, and that our F1 (0.87) is above the paper's (0.84) while accuracy and MCC are below.

How we checked it: the code cells and all their outputs are identical to the version before the rewrite (compared cell by cell). Every number in the new texts was searched in the outputs of the code cells; the few that are not printed are percentages of printed values, the 80/20 split chosen in the code, the number of columns derived from the printed shapes, or a reference year.
