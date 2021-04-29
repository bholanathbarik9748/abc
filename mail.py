import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

# Graph initialization
df = pd.read_csv("home_dic.csv")
plt.xlabel('area(sq ft)')
plt.ylabel('price')
plt.scatter(df.area, df.price, color='Green', marker='*')

# train Model
reg = linear_model.LinearRegression()
reg.fit(df[['area']], df.price)

# predict value
p_value = reg.predict([[3300]])
plt.plot(df.area, reg.predict(df[['area']]), color='red')
plt.show()
