### Best Results

| Model | AUC-PR (↑) | FNR (↓) | IoU (↑) | Fire Front MAE (↓) |
|:---|:---|:---|:---|:---|
| MLPCNN BCE | 0.0076 | 0.9975 | 0.0018 | 29.62 |
| MLPCNN Weighted BCE | 0.0078 | 0.9568 | 0.0069 | 22.81 |
| MLPCNN Focal | 0.0029 | 1.0000 | 0.0000 | NaN |
| CAE | 0.0096 | 0.9137 | 0.0110 | 33.31 |
| CAE (more training) | 0.0122 | 0.6415 | 0.0126 | 28.87 |
| NDWS_CAE (reported) | 0.284 | N/A | N/A | N/A |
| NDWS_CAE (our implementation) | 0.0097 | 0.9975 | 0.0007 | 49.2 |

Run `./job.sh` to run `train.py` and start a training job (all parameters are inside `job.sh` and all code is within `train.py` and `functions.py`)
Run `./eval_job.sh` to run `eval.py` and start an eval job (update parameters/pathways inside `eval_job.sh` and all code is within `eval.py` and `functions.py`)

### Conda env maintenance

To activate the env:

```
conda activate cvse
```

To recreate the environment from scratch:

```
conda env create -f environment.yml
```

To update the current environment using the environment.yml file

```
conda env update --name cvse --file environment.yml --prune
```


To update the environment.yml file:

```
conda env export > environment.yml
```

if any issues with any of this, you might have to open up ```environment.yml``` and just remove the line (prefix) with someone else's machine's pathway hardcoded in. 

### Dataset Preprocessing/Accessing

The original dataset can be downloaded from Kaggle (https://www.kaggle.com/datasets/fantineh/next-day-wildfire-spread?resource=download) then add the .tfrecord files into a folder called "data" in this directory. Then the first few cells of the ```evaluation.ipynb``` notebook show how to convert those into a dataset using details from here (https://www.kaggle.com/code/leelacauelas/wildfire-personal-copy-everything). 

The modified dataset is available on Google Cloud. The code to generate it can be found in ```modified_data_export```. The original code to create the dataset was copied from the Google Research github (https://github.com/google-research/google-research/tree/master/simulation_research/next_day_wildfire_spread). Lines marked "#ADDED" are the changes. To recompile the dataset via Google Earth Engine with the added forecast data and expanded date range, run:

```cd ./modified_data_export/next_day_wildfire_spread```
```pip3 install -r requirements.txt```
```cd ../```
```python3 -m simulation_research.next_day_wildfire_spread.data_export.export_ee_training_data_main --bucket=[BUCKET_NAME (for our cloud: ndws_modded)] --start_date="YYYY-MM-DD"[earliest: 2015-07-01] --end_date="YYYY-MM-DD"[latest: 2023-02-16]```

Note that lines 68, 69, and 70 of ```export_ee_training_data_main.py``` must be modified to set up your authorization for your specific Google Cloud project and bucket. This will include setting up a service account and download the JSON private key.
