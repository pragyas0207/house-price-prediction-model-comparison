import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

file_path="datasets/melb_data.csv"
mel_data=pd.read_csv(file_path)
print(mel_data.describe)
y=mel_data.Price
# features=



def get_mae(train_X,val_X,train_y,val_y):
    model=RandomForestRegressor(random_state=1)
    model.fit(train_X,train_y)
    preds=model.predict(val_X)
    return mean_absolute_error(val_y,preds)

# Impute the missing vals