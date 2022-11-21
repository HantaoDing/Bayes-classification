# Bayes decision
 
## 1. Introduction

This is a simple binary classification task based on Bayes decision, using the GaussianNB classifier to classify ECG timing signals, and then using Neyman-Pearson Classification to classify the above problems again

## 2. Environment

python==3.9.7

pytorch==1.11.0

numpy==1.21.6

## 3. Dataset

TwoLeadECG is an ECG data set taken from Physionet [1] and formatted by Eamonn Keogh. The data are originally from MIT-BIH Long-Term ECG Database (ltdb) Record ltdb/15814, begin at time 420 and end at 1019. The task is to distinguish between signal 0 and signal 1.

Train size: 23

Test size: 1139

Missing value: No

Number of classses: 2

Time series length: 82

Data donated by Eamonn Keogh (see [1], [2]).

## 4. Train

### 4.1 Data preprocessing

Data preprocessing is divided into two aspects, one does not perform feature engineering on the original data, and directly enters it into the classifier; The other is inputing all the data into the Tsfresh feature extraction tool in Python to extract features, and input the extracted features into the classifier.

If your Python does not have Tsfresh installed, please do the following

```sh
pip install tsfresh

```

### 4.2 Performance

The classification results based on GaussianNB classifier and Neyman-Pearson Classification are as follows

#### 4.2.1 GaussianNB

|              Data preprocessing              |     test acc     |
| :--------------------------------------: | :---------: |
| None |    0.70   |
|    Feature extraction     | 0.48 |
| Feature extraction and feature selection  | 0.89 |

**Note**:The reason for the decrease in accuracy after feature extraction may be that there are too many features extracted, and some useless features will cause the performance of the classifier to degrade. After feature selection, some useless features are discarded, and the final result is significantly improved

#### 4.2.2 Neyman-Pearson Classification

|              Data preprocessing              |     test acc     |
| :--------------------------------------: | :---------: |
| None |    0.75   |
|    Feature extraction     | 0.48 |
| Feature extraction and feature selection  | 0.84 |

The classifier parameters are shown in the following table

|              parameter              |     value     |
| :--------------------------------------: | :---------: |
| model |    logistic   |
|    alpha     | 0.45 |
| delta  | 0.3 |
| split | 1 |
| split_ratio | 0.5 |
| n_cores | 1 |
| band | False |
| rand_seed | 0 |

[1] https://www.physionet.org/

[2] http://www.timeseriesclassification.com/description.php?Dataset=TwoLeadECG

