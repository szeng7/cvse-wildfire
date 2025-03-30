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

Download the dataset from Kaggle (https://www.kaggle.com/datasets/fantineh/next-day-wildfire-spread?resource=download) then add the .tfrecord files into a folder called "data" in this directory. Then the first few cells of the ```evaluation.ipynb``` notebook show how to convert those into a dataset using details from here (https://www.kaggle.com/code/leelacauelas/wildfire-personal-copy-everything). There's capabilities to pull things directly from Google Earth Engine which we should look into at some point (https://github.com/google-research/google-research/tree/master/simulation_research/next_day_wildfire_spread#data-export) -- can pull from specific date ranges with this. 
