# Baseline (independence testing)

Uses the chi-square test (https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.chi2.html) to create a distance table for the tasks.

## Usage:

1. Activate the Python environment for this repository

2. Run the Python code in `causal_distances/BASELINE/chi2_baseline.py`, giving the prefix for the output as a command line argument `outprefix`, e.g., 

```
python3 ./causal_distances/BASELINE/chi2_baseline.py --outprefix /path/to/outdir/prefix
```

this will save the output in the file `{outprefix}_causal_distance_BASELINE_method.csv`
