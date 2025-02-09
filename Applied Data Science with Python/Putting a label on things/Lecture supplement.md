# Receiver operating characteristic curve

## ROC curve

An **ROC curve** (**receiver operating characteristic curve**) is a graph showing the performance of a classification model at all classification thresholds. This curve plots two parameters:

- True Positive Rate
- False Positive Rate

**True Positive Rate** (**TPR**) is a synonym for recall and is therefore defined as follows:

���=����+��

**False Positive Rate** (**FPR**) is defined as follows:

���=����+��

An ROC curve plots TPR vs. FPR at different classification thresholds. Lowering the classification threshold classifies more items as positive, thus increasing both False Positives and True Positives. The following figure shows a typical ROC curve.

[![](https://developers.google.com/static/machine-learning/crash-course/images/ROCCurve.svg)](https://developers.google.com/static/machine-learning/crash-course/images/ROCCurve.svg)

**Figure 4. TP vs. FP rate at different classification thresholds.**

To compute the points in an ROC curve, we could evaluate a logistic regression model many times with different classification thresholds, but this would be inefficient. Fortunately, there's an efficient, sorting-based algorithm that can provide this information for us, called AUC.

## AUC: Area Under the ROC Curve

**AUC** stands for "Area under the ROC Curve." That is, AUC measures the entire two-dimensional area underneath the entire ROC curve (think integral calculus) from (0,0) to (1,1).

[![](https://developers.google.com/static/machine-learning/crash-course/images/AUC.svg)](https://developers.google.com/static/machine-learning/crash-course/images/AUC.svg)

**Figure 5. AUC (Area under the ROC Curve).**

AUC provides an aggregate measure of performance across all possible classification thresholds. One way of interpreting AUC is as the probability that the model ranks a random positive example more highly than a random negative example. For example, given the following examples, which are arranged from left to right in ascending order of logistic regression predictions:

[![](https://developers.google.com/static/machine-learning/crash-course/images/AUCPredictionsRanked.svg)](https://developers.google.com/static/machine-learning/crash-course/images/AUCPredictionsRanked.svg)

**Figure 6. Predictions ranked in ascending order of logistic regression score.**

AUC represents the probability that a random positive (green) example is positioned to the right of a random negative (red) example.

AUC ranges in value from 0 to 1. A model whose predictions are 100% wrong has an AUC of 0.0; one whose predictions are 100% correct has an AUC of 1.0.

AUC is desirable for the following two reasons:

- AUC is **scale-invariant**. It measures how well predictions are ranked, rather than their absolute values.
- AUC is **classification-threshold-invariant**. It measures the quality of the model's predictions irrespective of what classification threshold is chosen.

However, both these reasons come with caveats, which may limit the usefulness of AUC in certain use cases:

- **Scale invariance is not always desirable.** For example, sometimes we really do need well calibrated probability outputs, and AUC won’t tell us about that.
- **Classification-threshold invariance is not always desirable.** In cases where there are wide disparities in the cost of false negatives vs. false positives, it may be critical to minimize one type of classification error. For example, when doing email spam detection, you likely want to prioritize minimizing false positives (even if that results in a significant increase of false negatives). AUC isn't a useful metric for this type of optimization.

# Boosted trees and gradient boosting

> [!info] A Visual Guide to Gradient Boosted Trees  
> An intuitive visual guide and video explaining GBT and the MNIST database  
> [https://towardsdatascience.com/a-visual-guide-to-gradient-boosted-trees-8d9ed578b33](https://towardsdatascience.com/a-visual-guide-to-gradient-boosted-trees-8d9ed578b33)  

> [!info] Introduction to Boosted Trees — xgboost 2.0.3 documentation  
> XGBoost stands for “Extreme Gradient Boosting”, where the term “Gradient Boosting” originates from the paper Greedy Function Approximation: A Gradient Boosting Machine, by Friedman.  
> [https://xgboost.readthedocs.io/en/stable/tutorials/model.html](https://xgboost.readthedocs.io/en/stable/tutorials/model.html)