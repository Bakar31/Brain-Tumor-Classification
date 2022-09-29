# Brain Tumor Classification

### Prediction of the status of a genetic biomarker important for brain cancer treatment.

## Goal:
To minimize the number of surgeries and refine the type of therapy required

<br>

## Motivation
Currently, genetic analysis of cancer requires surgery to extract a tissue sample. Then it can take several weeks to determine the genetic characterization of the tumor. Depending upon the results and type of initial therapy chosen, a subsequent surgery may be necessary. If an accurate method to predict the genetics of the cancer through imaging (i.e., radiogenomics) alone could be developed, this would potentially minimize the number of surgeries and refine the type of therapy required.

## Description:
Data are structured as follows: 
Each independent case has a dedicated folder identified by a five-digit number. Within each of these “case” folders, there are four sub-folders, each of them corresponding to each of the structural multi-parametric MRI (mpMRI) scans, in DICOM format. The exact mpMRI scans included are:

<li> Fluid Attenuated Inversion Recovery (FLAIR) </li>
<li> T1-weighted pre-contrast (T1w) </li>
<li> T1-weighted post-contrast (T1Gd) </li>
<li> T2-weighted (T2) </li>

<br>

## Files:
<li> <b>train/</b> - folder containing the training files, with each top-level folder representing a subject. </li>
<li> <b>train_labels.csv</b>  - file containing the target MGMT_value for each subject in the training data (e.g. the presence of MGMT promoter methylation). </li>
<li> <b>test/</b>  - the test files, which use the same structure as <b>train/</b>. </li>
<li> <b>sample_submission.csv</b>  - a sample submission file in the correct format. </li>
