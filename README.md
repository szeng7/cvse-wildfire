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