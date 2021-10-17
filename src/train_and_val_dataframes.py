import os
import pandas as pd
from data import df, df_test

def get_train_val_dataframe(mri_type):
    all_img_files = []
    all_img_labels = []
    all_img_patient_ids = []
    for row in df.iterrows():
        if row[1]['BraTS21ID_full'] == '00109' and mri_type == 'flair':
            continue
        if row[1]['BraTS21ID_full'] == '00123' and mri_type == 't1w':
            continue
        if row[1]['BraTS21ID_full'] == '00709' and mri_type == 'flair':
            continue
        img_dir = row[1][mri_type]
        img_files = os.listdir(img_dir)
        img_nums = sorted([int(ele.replace('Image-', '').replace('.dcm', '')) for ele in img_files])
        mid_point = int(len(img_nums)/2)
        start_point = mid_point - max(int(mid_point*0.1), 1)
        end_point = mid_point + max(int(mid_point*0.1), 1)
        img_names = [f'Image-{img_nums[i]}.dcm' for i in range(start_point, end_point+1)]
        img_paths = [img_dir+ele for ele in img_names]
        img_labels = [row[1]['MGMT_value']]*len(img_paths)
        img_patient_ids = [row[1]['BraTS21ID']]*len(img_paths)
        all_img_files.extend(img_paths)
        all_img_labels.extend(img_labels)
        all_img_patient_ids.extend(img_patient_ids)

    train_val_df = pd.DataFrame({'patient_ids': all_img_patient_ids,
                  'labels': all_img_labels,
                  'file_paths': all_img_files})

    train_val_df['labels'] = train_val_df['labels'].map({1: '1', 0: '0'})
    
    return train_val_df
    
def get_test_dataframe(mri_type):   
    all_test_img_files = []
    all_test_img_labels = []
    all_test_img_patient_ids = []
    for row in df_test.iterrows():
        img_dir = row[1][mri_type]
        img_files = os.listdir(img_dir)
        img_nums = sorted([int(ele.replace('Image-', '').replace('.dcm', '')) for ele in img_files])
        mid_point = int(len(img_nums)/2)
        start_point = mid_point - max(int(mid_point*0.1), 1)
        end_point = mid_point + max(int(mid_point*0.1), 1)
        img_names = [f'Image-{img_nums[i]}.dcm' for i in range(start_point, end_point+1)]
        img_paths = [img_dir+ele for ele in img_names]
        img_labels = [row[1]['MGMT_value']]*len(img_paths)
        img_patient_ids = [row[1]['BraTS21ID']]*len(img_paths)
        all_test_img_files.extend(img_paths)
        all_test_img_labels.extend(img_labels)
        all_test_img_patient_ids.extend(img_patient_ids)

    test_df = pd.DataFrame({'patient_ids': all_test_img_patient_ids,
                  'labels': all_test_img_labels,
                  'file_paths': all_test_img_files})
    
    test_df['labels'] = ['1']*(len(test_df)-1) + ['0'] # workaround for testing data gen
    
    return test_df

# train dataframes
train_df_flair = get_train_val_dataframe('flair')
train_df_t1w = get_train_val_dataframe('t1w')
train_df_t1wce = get_train_val_dataframe('t1wce')
train_df_t2w = get_train_val_dataframe('t2w')

# test dataframes
test_df_flair = get_test_dataframe('flair')
test_df_t1w = get_test_dataframe('t1w')
test_df_t1wce = get_test_dataframe('t1wce')
test_df_t2w = get_test_dataframe('t2w')