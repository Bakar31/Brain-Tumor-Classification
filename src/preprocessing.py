from data import *

# full paths for each id for different types of sequences
def full_ids(data):
    zeros = 5 - len(str(data))
    if zeros > 0:
        prefix = ''.join(['0' for i in range(zeros)])
    
    return prefix+str(data)
        

df['BraTS21ID_full'] = df['BraTS21ID'].apply(full_ids)
df_test['BraTS21ID_full'] = df_test['BraTS21ID'].apply(full_ids)

# adding all the paths to the train df for easy access
df['flair'] = df['BraTS21ID_full'].apply(lambda file_id : root_dir+'train/'+file_id+'/FLAIR/')
df['t1w'] = df['BraTS21ID_full'].apply(lambda file_id : root_dir+'train/'+file_id+'/T1w/')
df['t1wce'] = df['BraTS21ID_full'].apply(lambda file_id : root_dir+'train/'+file_id+'/T1wCE/')
df['t2w'] = df['BraTS21ID_full'].apply(lambda file_id : root_dir+'train/'+file_id+'/T2w/')


# adding all the paths to the test df for easy access
df_test['flair'] = df_test['BraTS21ID_full'].apply(lambda file_id : root_dir+'test/'+file_id+'/FLAIR/')
df_test['t1w'] = df_test['BraTS21ID_full'].apply(lambda file_id : root_dir+'test/'+file_id+'/T1w/')
df_test['t1wce'] = df_test['BraTS21ID_full'].apply(lambda file_id : root_dir+'test/'+file_id+'/T1wCE/')
df_test['t2w'] = df_test['BraTS21ID_full'].apply(lambda file_id : root_dir+'test/'+file_id+'/T2w/')