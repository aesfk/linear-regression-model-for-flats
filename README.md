# Least-Squares Regression for Moscow Flat Prices

A linear algebra lab that fits ordinary least-squares models in NumPy and measures how feature engineering changes prediction error on a Moscow flat-price dataset. The same [notebook](homework_practice_02_linregr_Emelchenkova.ipynb) also explores polynomial overfitting, L2 regularization and incremental least-squares updates on a simulated stream.

## Moscow flat-price experiment

The included dataset has 1,530 rows and five predictors covering floor areas and distance. The notebook uses a reproducible 70/30 split with `random_state=179`. The extended model adds kitchen share, an interaction between total floor area and distance, and log distance.

| Model | Test MSE |
| --- | ---: |
| Predict the training-set mean price | 2,346.03 |
| Ordinary least squares with the original predictors | 761.67 |
| Ordinary least squares with engineered features | 676.49 |

The table uses the same 459 test rows for all three models. MSE is in squared dataset target units. The mean-price row provides a reference for the two fitted models.

## Reproduce the housing experiment

From the repository root, install the dependencies and start Jupyter.

```bash
python -m pip install notebook numpy matplotlib sympy scikit-learn
python -m jupyter notebook
```

Open the notebook and go to Task 2, the Moscow flat-price experiment. In its first code cell, replace the Google Drive path with `Копия flats_moscow_mod.txt` and keep the tab delimiter and header skip. Run the code cells in Task 2 in order. The training-mean reference in the table is `numpy.mean((y_test - y_train.mean()) ** 2)`.

The polynomial and regularization exercises use a separate pair of one-dimensional course datasets. The repository includes `Копия train.txt`. For an end-to-end local run, obtain the course test file, `Копия test.txt`, skip the Google Drive mount cell and update the training and test paths in those sections. The notebook preview includes saved plots.
