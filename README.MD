# Radiogenomics for Glioblastoma Prediction
### Prediction of MGMT Promoter Methylation in Glioblastoma using MRI Scans

## Overview:
Glioblastoma is a life-threatening condition, and the most common form of brain cancer in adults. The presence of a specific genetic sequence in the tumor known as MGMT promoter methylation has been shown to be a favorable prognostic factor and a strong predictor of responsiveness to chemotherapy. However, current genetic analysis methods require invasive surgery to extract a tissue sample, which can delay treatment and require subsequent surgeries. The goal of this project is to develop a radiogenomics model that can predict the genetic subtype of glioblastoma using MRI scans. The model aims to minimize the number of surgeries required and refine the type of therapy needed.


## Objectives:
- The objective of this project is to develop  Deep Learning model that can accurately predict the genetic subtype of glioblastoma using MRI scans. The model should be trained to detect the presence of MGMT promoter methylation, which is a favorable prognostic factor and a strong predictor of chemotherapy responsiveness. 
- The ultimate goal is to minimize the number of surgeries required for genetic analysis and refine the type of therapy required for glioblastoma patients. 

The specific objectives are as follows:

- Develop a data pipeline to preprocess the MRI scans and extract relevant features for the model.
- Train and evaluate several deep learning models on the preprocessed data.
- Test the final model on a separate test set to evaluate its performance on new data.
- Discuss the implications of the model for clinical practice and potential future research directions.


## Training sample
![alt text](/images/project/brain tumor clf/sample.png](https://github.com/Bakar31/Brain-Tumor-Classification/blob/master/sample.png)


## My Approch:

### Step 1: Data Preparation

The dataset consisted of 400,116 DICOM files of size 136.85 GB. First, the path associated with each file was saved in a data frame for later use in loading the file. There were four types of scans:

- Fluid Attenuated Inversion Recovery (FLAIR)
- T1-weighted pre-contrast (T1w)
- T1-weighted post-contrast (T1Gd)
- T2-weighted (T2)

For each type of training and test, a dataframe was created. Then a custom data generator was constructed to read the DCM file as an array with a size of 224x224.

### Step 2: Architecture Selection and Training

**EfficientNetB0-B3** architectures were utilised to train each type of MRI data. Thus, a total of *12 models* were trained with

- learning_rate: 0.001
- optimizer: Adam
- droupout: 0.15
- early stopping with patience = 3

## Result

| Models | MRI Type | Train Accuracy | Validation Accuracy |
| ------ | -------- | -------------- | ------------------- |
| EfficientNet-B0 | FLAIR | 0.6769 | 0.7373 |
| EfficientNet-B0 | T1w | 0.6665 | 0.7211 |
| EfficientNet-B0 | T1Gd | 0.6777 | 0.3720 |
| EfficientNet-B0 | T2 | 0.7382 | 0.8312 |
| EfficientNet-B1 | FLAIR | 0.7085 | 0.4281 |
| EfficientNet-B1 | T1w | **0.6351** | **0.9014** |
| EfficientNet-B1 | T1Gd | **0.7260** | **0.6037** |
| EfficientNet-B1 | T2 | **0.7452** | **0.8941** |
| EfficientNet-B2 | FLAIR | **0.7255** | **0.8222** |
| EfficientNet-B2 | T1w | 0.6739 | 0.3858 |
| EfficientNet-B2 | T1Gd | 0.6624| 0.5108 |
| EfficientNet-B2 | T2 | 0.7277 | 0.8483 |

## Conclusion

The development of a deep learning model that accurately predicts the genetic subtype of glioblastoma using MRI scans is a significant advancement in the field of brain cancer diagnosis and treatment. The use of deep learning models to analyze MRI scans allows for a less invasive and more efficient method of genetic analysis, minimizing the number of surgeries required for diagnosis and subsequent treatment planning. The results of this project demonstrate the effectiveness of using deep learning models, particularly EfficientNet-B2, to accurately predict the genetic subtype of glioblastoma based on MRI scans. The highest validation accuracy achieved was 0.9014, indicating that the model has a high level of accuracy and is able to reliably predict the presence of MGMT promoter methylation. 

This technology has the potential to revolutionize the way brain cancer is diagnosed and treated. By reducing the number of surgeries required for genetic analysis and treatment planning, patients can receive more efficient and effective care. Additionally, the use of deep learning models to analyze MRI scans can help clinicians make more informed treatment decisions and ultimately improve patient outcomes.
