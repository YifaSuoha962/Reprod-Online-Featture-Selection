# Online Feature Selection (Reproduced)

This work is a re-implemented version for the work **A novel framework for online supervised learning
with feature selection**. The original code is available at [OFSA](https://github.com/barbua/OFSA), we thank the authors' great contribution.

## Environment Preparation

```console
conda create -n ofsa python=3.8 
pip install -r requirements.txt
```

## Regression on simulated data
Run ```python comparion_FSA_Lasso_reg.py``` for online lasso and feature selection.
Run ```python offline_methods_reg.py``` for offline lasso and feature selection.

## Regression on Wikiface dataset

Open the jupyter  notebook file `wikiface_test.ipynb` and run cell-by-cell.
