# MR analysis

Usage:

1. **R setup (tested for R version 4.3.0)**

```
install.packages("data.table", dependencies=TRUE)
install.packages("lubridate", dependencies=TRUE)
install.packages("dplyr", dependencies=TRUE)
install.packages("readxl", dependencies=TRUE)
install.packages("tidyverse") # needed to install Matrix v1.5-3, MASS v7.3-58.1, mgcv v1.8-41 to solve dependency conflict
install.packages("remotes", dependencies=TRUE) # may help to resolve dependency conflict
library(remotes)
install_github("MRCIEU/TwoSampleMR") # for MR
```

2. **Download auxiliary data for instruments**

Requires suitable instruments - see https://mrcieu.github.io/TwoSampleMR/articles/exposure.html#obtaining-instruments-from-existing-catalogues. We used GWAS summary statistics and fine-mapping analyses for the UK Biobank and FinnGen datasets (see the paper for details).

3. **MR analysis**

The `CausalMR.R` script shows an example of how we carried out an MR analysis, which could be used as a starting point to edit for your data and needs.

2. **Download auxiliary data for instruments**

Requires suitable instruments - see https://mrcieu.github.io/TwoSampleMR/articles/exposure.html#obtaining-instruments-from-existing-catalogues. We used GWAS summary statistics and fine-mapping analyses for the UK Biobank and FinnGen datasets (see the paper for details).

3. **MR analysis**

The `CausalMR.R` script shows an example of how we carried out an MR analysis, which could be used as a starting point to edit for your data and needs.

4. **Construct causal distance file from results of MR analysis**

To construct the causal distance file using Python (the environment for this repository), run `python mr.py` with the following arguments:
  - `--mr_filepattern` the glob pattern for your MR files
  - `--outprefix` a prefix for the output files

Example usage:
```
python3 ./causal_distances/MR/mr.py --mr_filepattern "/file/pattern/for/mr/files" --outprefix /path/to/outdir/prefix
```

This will generate two output files: `{outprefix}_causal_distance_MR_method.csv` and `{outprefix}_causal_estimates_MR_method.csv`, which contain the constructed causal distance file and causal estimates, respectively.
