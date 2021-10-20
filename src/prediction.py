import numpy as np
import pandas as pd
from data import root_dir
from data_generators import test_flair, test_t1w, test_t1wce, test_t2w, test_df_flair, test_df_t1wce, test_df_t1w, test_df_t2w
from model_efficientnet_B0 import efficientB0_flaire, efficientB0_t1w, efficientB0_t1wce, efficientB0_t2w
from model_efficientnet_B1 import efficientB1_flaire, efficientB1_t1w, efficientB1_t1wce, efficientB1_t2w

pred_values = []

# making prediction
def predictions(model, data, dataframe):
    test_pred = model.predict(data, steps=len(data))
    dataframe['pred_y'] = test_pred
    mean_pred = test_pred.mean()
    preds = dataframe.groupby('patient_ids').apply(lambda x: x['pred_y'].max()
        if (x['pred_y'].max() - mean_pred) > (mean_pred - x['pred_y'].min()) 
        else x['pred_y'].min())
    return preds.values

# flaire prediction
flaire1 = predictions(efficientB0_flaire, test_flair, test_df_flair)
flaire2 = predictions(efficientB1_flaire, test_flair, test_df_flair)

# tw1 prediction
t1w1 = predictions(efficientB0_t1w, test_t1w, test_df_t1w)
t1w2 = predictions(efficientB1_t1w, test_t1w, test_df_t1w)

# tw1ce prediction
t1wce1 = predictions(efficientB0_t1wce, test_t1wce, test_df_t1wce)
t1wce2 = predictions(efficientB1_t1wce, test_t1wce, test_df_t1wce)

# tw2 predicton
t2w1 = predictions(efficientB0_t2w, test_t2w, test_df_t2w)
t2w2 = predictions(efficientB1_t2w, test_t2w, test_df_t2w)

pred_values.append(flaire1)
pred_values.append(flaire2)
pred_values.append(t1w1)
pred_values.append(t1w2)
pred_values.append(t1wce1)
pred_values.append(t1wce2)
pred_values.append(t2w1)
pred_values.append(t2w2)


all_test_preds = np.array(pred_values)
sub = pd.read_csv(root_dir+'sample_submission.csv') 
sub['MGMT_value'] = all_test_preds.mean(0) 
sub.to_csv("submission.csv", index=False)