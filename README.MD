# Brain Tumor Classification

### Prediction of the status of a genetic biomarker important for brain cancer treatment.

## Goal:
To minimize the number of surgeries and refine the type of therapy required

## Motivation
Currently, genetic analysis of cancer requires surgery to extract a tissue sample. Then it can take several weeks to determine the genetic characterization of the tumor. Depending upon the results and type of initial therapy chosen, a subsequent surgery may be necessary. If an accurate method to predict the genetics of the cancer through imaging (i.e., radiogenomics) alone could be developed, this would potentially minimize the number of surgeries and refine the type of therapy required.

## Description:
Given data are structured as follows: 
Each independent case has a dedicated folder identified by a five-digit number. Within each of these “case” folders, there are four sub-folders, each of them corresponding to each of the structural multi-parametric MRI (mpMRI) scans, in DICOM format. The exact mpMRI scans included are:

<li> Fluid Attenuated Inversion Recovery (FLAIR) </li>
<li> T1-weighted pre-contrast (T1w) </li>
<li> T1-weighted post-contrast (T1Gd) </li>
<li> T2-weighted (T2) </li>

## My Approch
There were to total <b>400116</b>  files of Size <b>136.85 GB</b> and <b>Type dcm</b>  was given. 
A dedicated <b>DataFrameIterator</b> was constructed that reads the data dcm file as an array with a size of 224x224. 
The path associated with each file was saved in a dataframe, which was later utilized to load the file. I implemented EfficentNetB0-B3. 
Each architecture was trained for each type of MRI data. 

<br>
Total 12 models was trained. Models for each type of scan were then ensambled.</p>
