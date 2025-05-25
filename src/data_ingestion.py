import pandas as pd 
import os
def data_ingestion():
    base_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))

    train_path = os.path.join(base_dir, "data", "train.csv")
    df_train = pd.read_csv(train_path)

    test_path = os.path.join(base_dir, "data", "test.csv")
    df_test = pd.read_csv(test_path)
    submission_path=os.path.join(base_dir, "data", "sample_submission.csv")
    df_target=pd.read_csv(submission_path)
    print(df_train.shape)
    return df_train,df_test,df_target
data_ingestion()