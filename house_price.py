import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline  ---->This works only in ipynb...so in .py just do plt.show() at the end

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


file="datasets/melb_data.csv"
data=pd.read_csv(file)
print(data.describe())

y=data.Price
features=['BuildingArea','YearBuilt','Rooms','Distance']
X=data[features]
print(data.shape)
print(data.info)