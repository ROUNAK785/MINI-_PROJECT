import yfinance as yf #library jisse data nikaal paao noobdi!!
import pandas as pd #library jisse data ko pandas dataframe mein convert kar paao noobdi!!
import numpy as np #library jisse numerical operations kar paao noobdi!!
import matplotlib.pyplot as plt #library jisse data ko dekh ke easily samajh paao noobdi!!

from sklearn.model_selection import train_test_split #library jisse data ko training aur testing mein divide kar paao noobdi!!
from xgboost import XGBRegressor, XGBClassifier #library jisse XGBoost model banake stock price predict kar paao noobdi!!
from sklearn.metrics import mean_squared_error , accuracy_score #library jisse model ki performance evaluate kar paao noobdi!!!   

data = yf.download("RELIANCE.NS", start="2020-01-01") # yfinance API ke through "RELIANCE.NS" ka 2020 se lekar aaj tak ka live data download kar rahe hain.
#print(data.head()) top 5 rows of the data print kar rahe hain pandas mai sir padhaye the.

data['MA5'] = data['Close'].rolling(window=5).mean() #5 din ka moving average calculate kar rahe hain, jisse stock price ke short-term trend ko samajh paao noobdi, kab upar jaa raha hai kab neeche aa raha hai!!
data['MA10'] = data['Close'].rolling(window=10).mean() #10 din ka moving average calculate kar rahe hain, jisse stock price ke trend ko samajh paao noobdi,kab upar jaa raha hai kab neeche aa raha hai!!
data['MA20'] = data['Close'].rolling(window=20).mean() #20 din ka moving average calculate kar rahe hain, jisse stock price ke medium-term trend ko samajh paao noobdi,same pehle jaise!!
data['MA30'] = data['Close'].rolling(window=30).mean() #30 din ka moving average calculate kar rahe hain, jisse stock price ke trend ko samajh paao noobdi,kab upar jaa raha hai kab neeche aa raha hai!!
data['MA40'] = data['Close'].rolling(window=40).mean() #40 din ka moving average calculate kar rahe hain, jisse stock price ke trend ko samajh paao noobdi,kab upar jaa raha hai kab neeche aa raha hai!!
data['MA50'] = data['Close'].rolling(window=50).mean() #50 din ka moving average calculate kar rahe hain, jisse stock price ke long-term trend ko samajh paao noobdi,same pehle jaise!!
data['Target'] = data['Close'].pct_change(periods=1) #target variable create kar rahe hain, jisme next day ka closing price store kar rahe hain, jisse model ko train karne ke liye use karenge!!

data.dropna(inplace=True) #missing values ko drop kar rahe hain, jisse model ko train karne mein problem na ho noobdi!!

data['Target_Class'] = (data['Close'].shift(-5) > data['Close']).astype(int) #target variable ko binary class mein convert kar rahe hain, jisme 1 ka matlab hai stock price upar jaayega aur 0 ka matlab hai stock price neeche aaayega!!ismai (-5) ka matlab hai 5 din baad ka closing price, aur (>) ka matlab hai agar 5 din baad ka closing price current day ke closing price se bada hai to 1 assign karo warna 0 assign karo!!   

data['Target_Reg'] = data['Close'].pct_change().shift(-5) #target variable ko shift kar rahe hain, jisse model ko train karne ke liye use karenge!!iska matlab hai target variable ko 5 din pehle shift kar rahe hain!!

data.dropna(inplace=True) #missing values ko drop kar rahe hain, jisse model ko train karne mein problem na ho noobdi!!

x = data[['MA10','MA20','MA30','MA40','MA50','Target']] #features ko select kar rahe hain, 10 din ka moving average aur 50 din ka moving average  and target variable include kar rahe hain!!
y_class = data['Target_Class'] #target variable for classification model ko select kar rahe hain!!
y_reg = data['Target_Reg'] #target variable for regression model ko select kar rahe hain!!

x_train_class, x_test_class, y_train_class, y_test_class = train_test_split(x, y_class, test_size=0.2, random_state=42) #data ko training aur testing mein divide kar rahe hain classification model ke liye!!
x_train_reg, x_test_reg, y_train_reg, y_test_reg = train_test_split(x, y_reg, test_size=0.2, random_state=42) #data ko training aur testing mein divide kar rahe hain regression model ke liye!!

model_class = XGBClassifier(n_estimators=100, learning_rate=0.01, random_state=42, max_depth=5) #XGBoost classification model create kar rahe hain, jisme 100 trees use kar rahe hain!!
model_class.fit(x_train_class, y_train_class) #classification model ko training data pe fit kar rahe hain!!
model_reg = XGBRegressor(n_estimators=100, learning_rate=0.01, random_state=42, max_depth=5) #XGBoost regression model create kar rahe hain, jisme 100 trees use kar rahe hain!!
model_reg.fit(x_train_reg, y_train_reg) #regression model ko training data pe fit kar rahe hain!!

y_pred_class = model_class.predict(x_test_class) #classification model se predictions nikaal rahe hain!!
y_pred_reg = model_reg.predict(x_test_reg) #regression model se predictions nikaal rahe hain!!

accuracy = accuracy_score(y_test_class, y_pred_class) #classification model ki accuracy calculate kar rahe hain!!
mse_class = mean_squared_error(y_test_class, y_pred_class) #classification model ka mean squared error calculate kar rahe hain!!
mse_reg = mean_squared_error(y_test_reg, y_pred_reg) #regression model ka mean squared error calculate kar rahe hain!!

print(f"Classification Model Accuracy: {accuracy:.2f}") #classification model ki accuracy print kar rahe hain!!
print(f"Regression Model Accuracy: {accuracy:.2f}") #regression model ki accuracy print kar rahe hain, lekin regression model ke liye accuracy ka concept nahi hota hai, isliye yeh line galat hai, isko hata dena chahiye tha!!
print(f"Classification Model Mean Squared Error: {mse_class:.4f}") #classification model ka mean squared error print kar rahe hain, lekin classification model ke liye mean squared error ka concept nahi hota hai, isliye yeh line galat hai, isko hata dena chahiye tha!!
print(f"Regression Model Mean Squared Error: {mse_reg:.4f}") #regression model ka mean squared error print kar rahe hain!!

latest = x.tail(1) #latest data point ko select kar rahe hain, jisse future prediction ke liye use karenge!!

future_pred_class = model_class.predict(latest)[0] #classification model se future prediction nikaal rahe hain!!
future_pred_reg = model_reg.predict(latest)[0] #regression model se future prediction nikaal rahe hain!!
print(f"Future Price Movement Prediction (Classification): {'Up' if future_pred_class == 1 else 'Down'}") #future price movement prediction print kar rahe hain, jisme agar prediction 1 hai to "Up" print karenge aur agar prediction 0 hai to "Down" print karenge!!
print(f"Future Price Change Prediction (Regression): {future_pred_reg:.4f}") #future price change prediction print kar rahe hain, jisme predicted percentage change ko 4 decimal places tak print karenge!!

if (future_pred_class == 1) and (future_pred_reg > 0): #agar classification model ka prediction 1 hai aur regression model ka prediction positive hai to "Stock price is likely to go up" print karenge!!
    print("Stock price is likely to go up")
elif (future_pred_class == 0) and (future_pred_reg < 0): #agar classification model ka prediction 0 hai aur regression model ka prediction negative hai to "Stock price is likely to go down" print karenge!!
    print("Stock price is likely to go down")

plt.figure(figsize=(14, 7)) #figure ka size set kar rahe hain!!
plt.plot(data['Close'], label='Close Price') #closing price ka line
plt.plot(data['MA5'], label='MA5') #5 din ka moving average ka line plot kar rahe hain!!
plt.plot(data['MA10'], label='MA10') #10 din ka moving average ka line plot kar rahe hain!!
plt.plot(data['MA20'], label='MA20') #20 din ka moving average ka line plot kar rahe hain!!
plt.plot(data['MA30'], label='MA30') #30 din ka moving average ka line plot kar rahe hain!!
plt.plot(data['MA40'], label='MA40') #40 din ka moving average ka line plot kar rahe hain!!
plt.plot(data['MA50'], label='MA50') #50 din ka moving average ka line plot kar rahe hain!!
plt.title('Stock Price with Moving Averages') #plot ka title set kar rahe
plt.legend()
plt.show()

plt.savefig("C:\\Users\\Sanjay Kumar Tiwary\\OneDrive\\Desktop\\MINI PROJECT\\stock_price_moving_averages.png") #plot ko "MINI PROJECT" folder mein "stock_price_moving_averages.png" naam se save kar rahe hain!!

import os
import joblib #library jisse model ko save kar paao noobdi!!
os.makedirs("C:\\Users\\Sanjay Kumar Tiwary\\OneDrive\\Desktop\\MINI PROJECT\\model", exist_ok=True)
joblib.dump(model_class, "C:\\Users\\Sanjay Kumar Tiwary\\OneDrive\\Desktop\\MINI PROJECT\\model\\xgb_classification_model.pkl") #classification model ko "model" folder mein save kar rahe hain!!
joblib.dump(model_reg, "C:\\Users\\Sanjay Kumar Tiwary\\OneDrive\\Desktop\\MINI PROJECT\\model\\xgb_regression_model.pkl") #regression model ko "model" folder mein save kar rahe hain!!
