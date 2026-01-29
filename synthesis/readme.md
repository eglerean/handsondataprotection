# Exercise on tabular data synthesis with Synthetic Data Vault (SDV)

Here a simple exercises that loads a sensitive spreadsheet (the one we use here for demos is not sensitive) and tries to synthesise a similar file so that statistical properties are kept similar, but the synthetic individuals do not exist.

*Your task is to first upload the xlsx or csv data that is available in the example folder. Note that you need to pseudonymised first using "suppression" as technig (= get rid of the columns that have direct identifiers). Then you run two python scripts, one to synthesise the dataset, another to verify the quality of the synthetic dataset.

## How to run

1. Make sure you have all the packages. One way to do this is to use conda/mamba (see the yaml file in the repository) or apptainer/singularity
	**You don't need to worry about this step with SD Desktop**.
2. Synthesise the data
	- Upload the code or copy paste it into vscode. 
	- The first code is `synthesize_sdv.py` which generates a synthetic spreadsheet as similar as possible to the original one.
	- You run the file from a terminal with `python3 synthesize_sdv.py`.
3. Compare the distribution of some of the variables with the script  `compare_real_vs_synth.py`


## To-do

* How does one make sure that the synthetic data is actually safe from privacy perspective?

