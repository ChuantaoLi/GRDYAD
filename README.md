# Dataset

The experiment is divided into two parts, the first part is validated on the cardiovascular disease datasets, and the second part is validated on the datasets from KEEL database. The cardiovascular disease datasets includes four: Survival7Y and Survival10Y are real follow-up datasets of ischemic heart disease provided by CNR Institute of Clinical Physiology in Italy, which are derived from: https://github.com/orientino/ml4cad. SPECTF Heart dataset from UCI database, available at https://archive.ics.uci.edu/dataset/96/spectf+heart. Framingham dataset from the Kaggle, available at https://www.kaggle.com/datasets/aasheesh200/framingham-heart-study-dataset. In the KEEL database, this study chose eight binary classification datasets, are available at https://sci2s.ugr.es/keel/imbalanced.php?order=name#sub10. An overview of these datasets is given in the following table.

|   Dataset    | Instance size | Feature size | Imbalance ratio |
| :----------: | :-----------: | :----------: | :-------------: |
|  Survival7Y  |     3987      |      18      |      5.34       |
| Survival10Y  |     3987      |      18      |      4.51       |
|    SPECTF    |      267      |      22      |      3.85       |
|  Framingham  |     4240      |      15      |      5.58       |
| ecoli-0_vs_1 |      220      |      7       |      1.86       |
|    ecoli1    |      336      |      7       |      3.36       |
|    ecoli2    |      336      |      7       |      5.46       |
|    ecoli3    |      336      |      7       |       8.6       |
|   haberman   |      306      |      3       |      2.78       |
| newthyroid1  |      215      |      5       |      5.14       |
| newthyroid2  |      215      |      5       |      5.14       |
|    yeast3    |     1484      |      8       |       8.1       |

# Code

The experiment is divided into three parts: the ablation experiment, the comparison experiment, and the parameter sensitivity analysis. `Ablation.py` is the code for the ablation experiment, including `AdaBoost`, `Only Random Undersampled`, `Only Dynamic Undersampled`, `Only SMOTE`, `Only Dynamic Oversampled`, `Random Undersampled + SMOTE`, and `Dynamic Oversampled + Dynamic Undersampled`. Among them, AdaBoost is implemented in the code of `AdaBoost.py`, and `Dynamic Oversampled + Dynamic Undersampled` is implemented in the code of `DSAD.py`. `Model_test.py` is the code of comparison experiment, which compares Logistic Regression, KNN, Decision Tree, Random Forest, Gradient Boosting, XGBoost, MLP, and AdaBoost, which are eight machine learning models, and again, AdaBoost is implemented in the `AdaBoost.py` code, and their iterations are set to 30 rounds as in GRDSAD. `GRDSAD.py` is the code for the models mentioned in this study. `d\eta\rho\d'_experiment1\2.py` These are the codes for the parameter-sensitive analysis.

# Parameterization

The experiments in the parameter sensitivity analysis section address four parameters: $\eta$, $\rho$, $\gamma$ and $d'$. $\eta$ denotes the degree of smoothing of the sigmoid function in the instance weight update, $\rho$ denotes the bandwidth of the heat kernel used to construct the similarity graph, $\gamma$ denotes the degree of regularization of the graph, and $d'$ denotes the $d'$ percent of components retained by the downscaling. The parameter settings for the 12 datasets used in the experiment are shown in the table below.

|   Dataset    | $\eta$ | $\rho$ | $\gamma$ | $d'$ |
| :----------: | :----: | :----: | :------: | :--: |
|  Survival7Y  |  0.1   |   1    |   0.01   | 30%  |
| Survival10Y  |  0.1   |   1    |   1.5    | 30%  |
|    SPECTF    |  0.1   |   1    |   0.01   | 40%  |
|  Framingham  |  0.4   |  1.1   |    3     | 50%  |
| ecoli-0_vs_1 |  0.1   |   1    |   0.01   | 80%  |
|    ecoli1    |  0.1   |   1    |   0.01   | 80%  |
|    ecoli2    |  0.5   |  1.2   |   0.1    | 50%  |
|    ecoli3    |  0.1   |   1    |   0.1    | 60%  |
|   haberman   |  0.4   |  1.4   |   0.1    | 40%  |
| newthyroid1  |  0.1   |   1    |   0.01   | 80%  |
| newthyroid2  |  0.1   |   1    |   0.05   | 80%  |
|    yeast3    |  0.5   |   2    |   0.5    | 30%  |

# Results

In the Results folder, `7Ydata.xlsx`, `10Ydata.xlsx`, `SPECTF.xlsx`, and `Framingham.xlsx` are the results of the Experiment1 section, which contains the results of the ablation experiments and the results of the comparison experiments. `KEEL.xlsx` is the results of the experiments for the eight test datasets, which include the results of the experiments with the performance comparison of seven mainstream Boosting variants, which are AdaM1, AdaM2, AdaNC9, LexiBoostM1, LexiBoostM2, DualLexiBoost, and JanEnsemble.The sources of these models are shown in the following table.

|                 Models                  |                            Papers                            |
| :-------------------------------------: | :----------------------------------------------------------: |
|              AdaM1, AdaM2               | Y. Freund and R. E. Schapire, “Experiments with a new boosting  algorithm,” in Proc. Int. Conf. Mach. Learn., 1996, pp. 148–156. |
|                 AdaNC9                  | S. Wang and X. Yao, “Multiclass imbalance problems: Analysis and  potential solutions,” IEEE Trans. Syst., Man, Cybern., Part B, vol. 42, no.  4, pp. 1119–1130, Aug. 2012. |
| LexiBoostM1, LexiBoostM2, DualLexiBoost | S. Datta, S. Nag, and S. Das, “Boosting with lexicographic  programming: Addressing class imbalance without cost tuning,” IEEE Trans.  Knowl. Data Eng., vol. 32, no. 5, pp. 883–897, May 2020. |
|               JanEnsemble               | Z. Jan, J. C. Munos, and A. Ali, “A novel method for creating an  optimized ensemble classifier by introducing cluster size reduction and  diversity,” IEEE Trans. Knowl. Data Eng., vol. 34, no. 7, pp. 3072–3081, Jul.  2022. |

In addition, `Heart parameter.xlsx` and `KEEL parameter.xlsx` contain the experimental results of the Parameter sensitivity analysis.
