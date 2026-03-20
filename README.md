1.PROJECT TITLE
    "Stock Market Prediction using XGBoost and Technical Indicators"

2.OBJECTIVE
    predict stock price direction(UP/DOWN) and precentage change for Reliance using historical data.

3.LIBRARIES USED
    yfinance,pandas,matplotlib,scikit-learn,xgboost

4.FEATURES
    moving averages:MA5,MA10,MA20,MA30,MA40,MA50
    return:daily precentage change

5.MODELS
    XGBoost Classifier:predicts direction
    XGBoost Regressor:predicts percentage change

6.RESULTS
    classification accuracy:63%
    regression mean squared error:0.0003
    Direction predictions uses both models: Strong/weak signal

7.FUTURE SCOPE
    predict for multiple stocks simultaneously
    build an web app for live prediction