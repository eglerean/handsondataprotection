# Exercise on tabular data synthesis with Synthetic Data Vault (SDV)

Here a simple exercises that loads a sensitive spreadsheet (the one we use here for demos is not sensitive) and tries to synthesise a similar file so that statistical properties are kept similar, but the synthetic individuals do not exist.

How to run

1. Make sure you have all the packages. One way to do this is to use conda/mamba or apptainer/singularity
2. Synthesise the data: run the script `synthesize_sdv.py` which generates a synthetic spreadsheet as similar as possible to the original one.
3. Compare the distribution of some of the variables with the script  `compare_real_vs_synth.py`


## To-do

* How does one make sure that the synthetic data is actually safe from privacy perspective?

