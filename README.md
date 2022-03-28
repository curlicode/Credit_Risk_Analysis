# Credit_Risk_Analysis

## Overview of the analysis
```Explain the purpose of this analysis.  The purpose of this analysis is well defined.```

The credit risk analysis challenge presents an unbalanced classification problem for covering the machine learning scope of data preparation, statistical reasoning, and model classification.  The credit risk of loans have an asymmetry in the amount of good to risky loans, thus offering the opportunity to use the `imbalanced-learn` and `scikit-learn` programming libraries to build and evaluate models.  Using credit card data classified as low- or high-risk, the `RandomOverSampler`, `SMOTE`, and `ClusterCentroids` algorithms are used to resample data with precision, accuracy, and recall to evaluate models, and the `BalancedRandomForestClassifier` and `EasyEnsembleClassifier` machine learning models are compared for assessing credit risk with reduced risk of bias.

## Results
```Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of all six machine learning models. Use screenshots of your outputs to support your results.  There is a bulleted list that describes the balanced accuracy score and the precision and recall scores of all six machine learning models (15 pt)```

* RandomOverSampler (oversample)
    * Confusion Matrix:

    || Predicted True | Predicted False |
    | --- | --- | --- |
    | Actually True | 44 (TP) | 44 (FN) |
    | Actually False | 4512 (FP) | 9164 (TN) |

    * Balanced Accuracy Score
      * 0.585
    * Precision (TP / (TP + FP))
      * high risk 0.01
      * low risk 1.00
    * Recall Scores (Sensitivity = TP / (TP + FN))
      * high risk 0.50
      * low risk  0.67
    * This model is rather hit-or-miss with a low F1 score with regards to identifying high risk loans.  High risk loans are a minority set within the data, so oversampling of the low risk loans with the selected algorithm has yielded overfitting and identifying a low risk more often than high risk.

* SMOTE (oversample)
  * Confusion Matrix:

  || Predicted True | Predicted False |
  | --- | --- | --- |
  | Actually True | 46 (TP) | 42 (FN) |
  | Actually False | 3794 (FP) | 9882 (TN) |

  * Balanced Accuracy Score
    * 0.623
  * Precision (TP / (TP + FP))
    * high risk 0.01
    * low risk 1.00
  * Recall Scores (Sensitivity = TP / (TP + FN))
    * high risk 0.52
    * low risk  0.72
  * This model also struggles with precision with a low F1 score with regards to identifying high risk loans.  High risk loans are a minority set within the data, so as with the previous algorithm, this algorithm also skews toward missing high risk loans in part because of the data composition, where the algorithm overly interpolates the majority data set.  SMOTE performs marginally better than RandomOverSampler.

* ClusterCentroids (undersample)
    * Confusion Matrix:

  || Predicted True | Predicted False |
  | --- | --- | --- |
  | Actually True | 47 (TP) | 41 (FN) |
  | Actually False | 6216 (FP) | 7460 (TN) |

  * Balanced Accuracy Score
    * 0.540
  * Precision (TP / (TP + FP))
    * high risk 0.01
    * low risk 0.99
  * Recall Scores (Sensitivity = TP / (TP + FN))
    * high risk 0.53
    * low risk  0.55
  * This model does not perform well compared with the prior algorithms, with a low accuracy and precision.  In undersampling, the algorithm has not successfully reduced the dataset in a balanced fashion when establishing the centroid aggregates.

* SMOTEENN (undersample)
    * Confusion Matrix:

  || Predicted True | Predicted False |
  | --- | --- | --- |
  | Actually True | 60 (TP) | 20 (FN) |
  | Actually False | 5093 (FP) | 8583 (TN) |

  * Balanced Accuracy Score
    * 0.655
  * Precision (TP / (TP + FP))
    * high risk 0.01
    * low risk 1.00
  * Recall Scores (Sensitivity = TP / (TP + FN))
    * high risk 0.68
    * low risk  0.63
  * The SMOTEENN algorithm yields better results with a higher accuracy.  It has limited precision like the other models, but model has a better ability to identify high risk loans even if at the expense of missing some high risk loans.  With sufficiently noisy data, the SMOTEENN model may also be vulnerable to undue effects of data in the model, which may be majority low risk.

* BalancedRandomForestClassifier (ensemble classifier)
    * Confusion Matrix:

  || Predicted True | Predicted False |
  | --- | --- | --- |
  | Actually True | 59 (TP) | 29 (FN) |
  | Actually False | 1571 (FP) | 12105 (TN) |

  * Balanced Accuracy Score
    * 0.778
  * Precision (TP / (TP + FP))
    * high risk 0.03
    * low risk 1.00
  * Recall Scores (Sensitivity = TP / (TP + FN))
    * high risk 0.70
    * low risk  0.87
  * The BalancedRandomForestClassifier algorithm yields better results with a higher accuracy and recall scores.  The method of balancing the data and undersampling yields a model that lessens the effects of the predominantly low risk records within the data.

* EasyEnsembleClassifier (ensemble classifier)
    * Confusion Matrix:

  || Predicted True | Predicted False |
  | --- | --- | --- |
  | Actually True | 75 (TP) | 13 (FN) |
  | Actually False | 702 (FP) | 12974 (TN) |

  * Balanced Accuracy Score
    * 0.900
  * Precision (TP / (TP + FP))
    * high risk 0.09
    * low risk 1.00
  * Recall Scores (Sensitivity = TP / (TP + FN))
    * high risk 0.92
    * low risk  0.94
  * The EasyEnsembleClassifier model yields robust matching rates.   

## Summary

```Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. If you do not recommend any of the models, justify your reasoning.  There is a summary of the results (2 pt).  There is a recommendation on which model to use, or there is no recommendation with a justification (3 pt)```

All of the models presented here are sensitive to the imbalance of credit risk data because the data inherently skews toward low risk.  Because of this differential, undersampling outperforms oversampling methods by better omitting excess and outlier low risk parts of the data.  However, given the complexity and number of features within the data and model, ensemble methods outperform the over- and under-sampling methods by transforming weak learners into a more robust model.  Although the approach resamples, thus still risks skewing toward the existing distribution within the data, `EasyEnsembleClassifier` is the recommended choice amongst these options because of its higher collective metrics with regards to Balanced Accuracy Scores, Precision, and Recall Scores.  Lastly, it is possible that the use of scalers may improve performance matches.
