import pandas as pd

root_dir = '../input/rsna-miccai-brain-tumor-radiogenomic-classification/'
df = pd.read_csv(root_dir+'train_labels.csv')
df_test = pd.read_csv(root_dir+'sample_submission.csv')