import pandas as pd 
import os
def data_ingestion():
    base_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))

    train_path = os.path.join(base_dir, "data", "train.csv")
    df_train = pd.read_csv(train_path)

    test_path = os.path.join(base_dir, "data", "test.csv")
    df_test = pd.read_csv(test_path)
    return df_train,df_test
if __name__=='__main__':
    data_ingestion()