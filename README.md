nlsub
==============================

Machine learning algorithm to remove non-linear noise in gravitational-wave data around GW200129. Full reproduction requires access to LIGO internal data and a GPU with about 30GB memory. Production-quality model training takes about 30 minutes on Nvidia A100 80GB.

For more scientific details, read [arXiv:2311.09921](https://arxiv.org/abs/2311.09921). 

Cleaned time-series data frame is publicly available on [Zenodo](https://zenodo.org/records/10143338).

![Image Alt text](/reports/figures/1264316116.345.png)

Examples of the cleaned data are in the `reports/figures` folder.

Installation
--------
1) Create a Conda environment with `make create_environment`
2) Install Python packages with `make requirements`
3) Install the `nlsub` package with `pip install .`

Makefile commands
--------
- `create_environment` - Set up Python interpreter environment.
- `test_environment` - Test Python environment setup.
- `get_data` - Download all required data. Requires access to LIGO internal data. Warning: some data is on tape, so this step may take hours.
- `whiten_data` - Whiten the time-series data.
- `features` - Prepare features for training the model.
- `train` - Train the model.
- `predict` - Clean the data around GW200129 using the model.
- `visualize` - Produce plots showing the difference between the original and cleaned data.
- `clean` - Delete all compiled Python files.
- `lint` - Lint using flake8.

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
