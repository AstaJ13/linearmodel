import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


csv_file_path = r'C:\Users\astaj\Downloads\archive\advertising.csv'

data= pd.read_csv(csv_file_path) #pandas (tabular data analysis)

print(data.head())  #1st 5 columns
print(data.columns)  #lists all column names

data= data.dropna(subset=['Daily Internet Usage'])

X=data[['Daily Internet Usage']]#indep variable (feature)
y= data['Area Income']          # dep variable (target)


X_train, X_test, y_train, y_test =train_test_split(X,y,test_size=0.2,random_state=50) 

model= LinearRegression()
model.fit(X_train,y_train)

y_pred= model.predict(X_test)

mse= mean_squared_error(y_test,y_pred)
print(f'Mean Squared Error",{mse:.2f}')

      
plt.scatter(X_test,y_test,color='blue',label='Actual Data')
plt.scatter(X_test,y_pred,color='red',label='Predicted Data')
plt.xlabel('Daily Internet Usage')
plt.ylabel('Area Income')
plt.legend()
plt.show()

