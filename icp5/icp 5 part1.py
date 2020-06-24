import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train = pd.read_csv('train.csv')

plt.scatter(train['SalePrice'],train['GarageArea'],color='red',alpha=.75)
filetr_data=(train['GarageArea']>200)&(train['GarageArea']<950)&(train['SalePrice']<500000)
print(train['GarageArea'],train['SalePrice'])
plt.ylabel('GarageArea')
plt.xlabel('SalePrice')




