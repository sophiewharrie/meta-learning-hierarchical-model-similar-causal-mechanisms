import random
from sklearn.preprocessing import StandardScaler
import pandas as pd

def get_train_val_test_data(df, NUM_DATASETS, INTERVENTIONS):
    
    NUMTASKS = len(df['task'].unique())
    NUMTASKS_VAL = round(NUMTASKS*0.15)
    NUMTASKS_TEST = NUMTASKS_VAL
    NUMTASKS_TRAIN = NUMTASKS - NUMTASKS_VAL - NUMTASKS_TEST

    # take minimum number of samples per task,
    # to ensure each task ends up with same amount of data (sampled at random)
    NUMSAMPLES_PER_TASK = df.groupby('task').count()['Y'].min()
    NUMSAMPLES_QRY = round(NUMSAMPLES_PER_TASK*0.5)
    NUMSAMPLES_SPT = NUMSAMPLES_PER_TASK - NUMSAMPLES_QRY

    # add extra columns for the intervention mask
    interv_cols_map = {'Y_interv':'Y'}
    df['Y_interv'] = 0
    for k in df.keys():
        if k.startswith('X_'): 
            interv_col_name = '{}_interv'.format(k)
            interv_cols_map[interv_col_name] = k
            if k in INTERVENTIONS:
                df[interv_col_name] = (df[k].astype(float)>0).astype(int) # assumes value of >0 indicates an intervention
            else:
                df[interv_col_name] = 0

    # scale the data 
    scaler = StandardScaler()
    interv_cols = list(interv_cols_map.keys()) # don't rescale intervention columns
    keys = df.drop(columns=['task']+interv_cols).keys()
    scaled_df = pd.DataFrame(scaler.fit_transform(df.drop(columns=['task']+interv_cols)), columns=keys)
    for col in ['task']+interv_cols: scaled_df.insert(loc=len(scaled_df.keys()), column=col, value=df[col].tolist())

    print("Making a dataset with a 70/15/15 train/val/test split\n")
    print(f"{NUMTASKS} tasks:")
    print(f"- {NUMTASKS_TRAIN} train tasks")
    print(f"- {NUMTASKS_VAL} val tasks")
    print(f"- {NUMTASKS_TEST} test tasks")
    print(f"{NUMSAMPLES_PER_TASK} samples per task:")
    print(f"- {NUMSAMPLES_SPT} support samples per task")
    print(f"- {NUMSAMPLES_QRY} query samples per task")

    TASKS = scaled_df['task'].unique().tolist()

    # make several datasets with tasks randomly assigned to train/val/test sets

    full_datasets = []
    full_interv_masks = []

    for dataset in range(1,NUM_DATASETS+1):

        random.shuffle(TASKS)
        train_tasks = TASKS[0:NUMTASKS_TRAIN]
        val_tasks = TASKS[NUMTASKS_TRAIN:NUMTASKS_TRAIN+NUMTASKS_VAL]
        test_tasks = TASKS[NUMTASKS_TRAIN+NUMTASKS_VAL:]

        meta_train_type = {'train':0, 'val':1, 'test':2}
        task_list = {'train':train_tasks, 'val':val_tasks, 'test':test_tasks}

        task_map = dict(zip(TASKS,range(len(TASKS)))) # relabel the tasks so they appear in-order in the dataset
        
        scaled_df = scaled_df.sample(frac=1)
        new_df = pd.DataFrame()
        for datatype in ['train', 'val', 'test']:
            add_data = scaled_df[scaled_df['task'].isin(task_list[datatype])]
            add_data.insert(loc=len(add_data.keys()), column='meta_train', value=meta_train_type[datatype])
            add_data.insert(loc=len(add_data.keys()), column='task_train', value=(add_data.groupby('task').cumcount() < NUMSAMPLES_SPT).astype(int).tolist())
            add_data['task'] = add_data['task'].map(task_map)
            new_df = pd.concat([new_df, add_data])

        new_df = new_df.sort_values(by=['task','task_train'])
        print(new_df.info())
        not_interv_cols = list(interv_cols_map.values())
        interv_cols = list(interv_cols_map.keys())

        full_datasets.append(new_df[not_interv_cols+['task','meta_train','task_train']])
        full_interv_masks.append(new_df[interv_cols+['task','meta_train','task_train']].rename(columns=interv_cols_map))

    return full_datasets, full_interv_masks
