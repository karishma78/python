from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split

train = pd.read_csv(Path('./winequality-red.csv'))
train.quality.describe()


n_features = train.select_dtypes(include=[np.number])
print('The top 3 correlated features \n')
corr=n_features.corr()
print(corr['quality'].sort_values(ascending=False)[:3], '\n')


nulls = pd.DataFrame(train.isnull().sum().sort_values(ascending=False))
nulls.columns = ['Null Count']
nulls.index.name = 'Feature'
print(nulls)

data = train.select_dtypes(include=[np.number]).interpolate().dropna()
print('The values without 0 sum',sum(data.isnull().sum() != 0))


y = np.log(train.quality)
X = data.drop(['quality'], axis=1)


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=.33)
model = lr.fit(X_train, y_train)



print("R^2 is: ", model.score(X_test, y_test))
predictions = model.predict(X_test)

from sklearn.metrics import mean_squared_error
print('RMSE is: ', mean_squared_error(y_test, predictions))

