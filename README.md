# Graph Regularized Dynamic Sampling AdaBoost

## Datasets

$\qquad$ In the `Experiment1` folder two real diagnostic return visit datasets for Ischemic Heart Disease provided by the CNR Institute of Clinical Physiology, Italy can be found. Survival7Y and Survival10Y represent the datasets for 7-and 10-year return visits, respectively. Description for the two datasets and source: https://github.com/orientino/ml4cad, SPECTF (Heart) dataset can be found: https://archive.ics.uci.edu/dataset/96/spectf+heart, Framingham dataset can be found: https://www.kaggle.com/datasets/aasheesh200/framingham-heart-study-dataset Furthermore, can be found in `Experiment2` folder KEEL database of eight binary classification datasets, they are the source of https://sci2s.ugr.es/keel/imbalanced.php?order=name#sub10.

## Code

$\qquad$ `Ablation_test.py` is the ablation experiment code for the IHD datasets. `AdaBoost.py` is the code for traditional AdaBoost. `DSAD.py` contains dynamic oversampling and dynamic undersampling code. `GRDSAD.py` is the Model code proposed in this study. `Model_test.py` is the code for the comparison experiment of multiple Machine Learning models on the IHD datasets. The Results of all experiments are recorded in `results.xlsx`, and `requirements.txt` is the version of the code import library.

