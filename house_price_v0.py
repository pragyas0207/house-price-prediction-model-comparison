import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

ny_file_path="datasets/melb_data.csv"
data=pd.read_csv(ny_file_path)
#data.describe()--->ipynb auto displays but .py doesnt so u hav to use print()
print(data.describe())

y=data.Price
feature_cols=['BuildingArea','YearBuilt','Rooms']
X=data[feature_cols]

melb_model=DecisionTreeRegressor(random_state=1)
melb_model.fit(X,y)
print("Original Values:")
print(y.head())
print("Predictions from same model(withput splitting): ")
print(melb_model.predict(X.head()))
print("*****MAE*****")
print(mean_absolute_error(y,melb_model.predict(X)))


# Splitting data into training and validation
# There's an error saying Nan Vals present if u do the below without preprocessing
print("\n-----------------Splitting Data (Decision Tree)------------------")
train_X,val_X,train_y,val_y=train_test_split(X,y,random_state=1)
model1=DecisionTreeRegressor(random_state=1)
model1.fit(train_X,train_y)
preds=model1.predict(val_X)
print(val_X.head())
print(pd.DataFrame(preds).head())
print("****MAE****")
print(mean_absolute_error(val_y,preds))


print("------------------Splitting Data (Random Forest)-----------------")
from sklearn.ensemble import RandomForestRegressor
model2=RandomForestRegressor(random_state=1)
model2.fit(train_X,train_y)
mod2_preds=model2.predict(val_X)
print(val_y.head())
print("Predictions: ",pd.DataFrame(mod2_preds).head())
print("*****MAE*****")
print(mean_absolute_error(val_y,mod2_preds))

# -------------------------MAE function---------------------
def get_mae(train_X,val_X,train_y,val_y):
    model=RandomForestRegressor(random_state=1)
    model.fit(train_X,train_y)
    prediction=model.predict(val_X)
    return mean_absolute_error(val_y,prediction)



print("*********************> PreProcessing <**************************")
print("Missing Vals:")
missing_val_count_by_column = (train_X.isnull().sum())
print(missing_val_count_by_column[missing_val_count_by_column > 0])


print("---------------- Drop Columns ---------------")
cols_with_missing=[cols for cols in train_X.columns
                   if train_X[cols].isnull().any()]
reduced_train_X=train_X.drop(cols_with_missing,axis=1)       #removed the NaN value cols from training data part of dataset
reduced_val_X=val_X.drop(cols_with_missing,axis=1)           #removed NaN value cols fron=m the validation data
print("MAE (Drop columns with missing values):")
print(get_mae(reduced_train_X,reduced_val_X,train_y,val_y))   #model created in get_mae function
# model3=RandomForestRegressor(random_state=1)
# model3.fit(reduced_train_X,train_y)
# pred3=model3.predict(reduced_val_X)
# print("MAE: ")
# print(mean_absolute_error(val_y,pred3))


print("--------------- Imputation ---------------------")
print("---***With MEAN***---")
from sklearn.impute import SimpleImputer
imputer=SimpleImputer()
imputed_train_X=pd.DataFrame(imputer.fit_transform(train_X))
imputed_val_X=pd.DataFrame(imputer.transform(val_X))
imputed_train_X.columns=train_X.columns
imputed_val_X.columns=val_X.columns
print("MAE (Imputed values): ")
print(get_mae(imputed_train_X,imputed_val_X,train_y,val_y))
print("---***With MEDIAN***---")
imputer=SimpleImputer(strategy="median")
imputed_train_X=pd.DataFrame(imputer.fit_transform(train_X))
imputed_val_X=pd.DataFrame(imputer.transform(val_X))
imputed_train_X.columns=train_X.columns
imputed_val_X.columns=val_X.columns
print("MAE (Imputed values): ")
print(get_mae(imputed_train_X,imputed_val_X,train_y,val_y))


print("\n-------------------------------------------------------")
print("-------------------------------------------------------")
print("Mean Absolte Errors: ")
print("Model= ",mean_absolute_error(y,melb_model.predict(X)))
print("Model1 (DecisionTree Split)= ",mean_absolute_error(val_y,preds)) #decisionTree split
print("Model2 (RandomForest Split)= ",mean_absolute_error(val_y,mod2_preds))    #RandomForest Split
print("Model3 (Drop cols)= ",get_mae(reduced_train_X,reduced_val_X,train_y,val_y))
print("Model4 (Imputation)= ",get_mae(imputed_train_X,imputed_val_X,train_y,val_y))
