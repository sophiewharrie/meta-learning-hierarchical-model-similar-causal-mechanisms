Scripts used for calculating similarity weights between tasks for the UK Biobank and FinnGen datasets, using 4 methods:

0. Baseline (CHI-2)
1. Mendelian randomisation (MR)
2. Invariant causal prediction (ICP)
3. Embedding of structural model (SCM/DAG method)

**Setup before running any method**: 
1. You will need the following data files prepared according to the format described in the main README.md file of this repository: (1) mainfile, (2) longfile, (3) metafile
1. Edit the filepaths and other global parameters at the top of the file `causal_distances/long_utils.py` for your data

**Instructions for each method**:

See the corresponding method directories for detailed instructions.
