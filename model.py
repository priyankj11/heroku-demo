import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
data = pd.read_csv('blasting_final_data.csv')
x = data.iloc[:, 6:23].values
y = data.iloc[:, 23].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=32)
regressor = LinearRegression()
regressor.fit(x_train, y_train)
regressor.fit([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],[0])
y_pred = regressor.predict(x_test)
pickle.dump(regressor, open('model.pkl', 'wb'))
