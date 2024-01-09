nlsub
==============================

Non-linear noise subtraction for gravitational-wave data tailored for GW200129 using machine learning. Full reproduction requires access to LIGO internal data and a GPU with about 30GB memory. Production-quality model training takes about 30 minutes on Nvidia A100 80GB.

For more scientific details, read [arXiv:2311.09921](https://arxiv.org/abs/2311.09921). 

Cleaned time-series data frame is publicly available on [Zenodo](https://zenodo.org/records/10143338).

Examples of the cleaned data are in `reports/figures` folder.

Makefile commands:
- `make get_data`           <- Download all required data. Requires access to LIGO internal data. Warning: data is on tape, so this step may take hours.
- `make whiten_data`        <- Whiten the time-series data.
- `make features`           <- Prepare features for the model training.
- `make train`              <- Train the model.
- `make predict`            <- Clean the data around GW200129 using the model 
- `make visualize`          <- Produce plots showing the difference between the original and cleaned data.
- `create_environment`      <- Set up Python interpreter evironment.
- `test_environment`        <- Test Python ennvironment setup
- `clean`                   <- Delete all compiled Python files.
- `lint`                    <- Lint using flake8.

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
