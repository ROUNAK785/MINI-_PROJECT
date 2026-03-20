from flask import Flask, render_template, request, redirect, session
import mysql.connector
import random
import smtplib
import bcrypt
import joblib
import pandas as pd
import yfinance as yf

import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET")

# ---------------- MYSQL CONNECTION ----------------
db = mysql.connector.connect(
    host=os.getenv("DB_HOST", "localhost"),
    user=os.getenv("DB_USER", "root"),
    password=os.getenv("DB_PASS"),
    database=os.getenv("DB_NAME", "user_auth")
)

cursor = db.cursor()

reg_model = joblib.load('model/xgb_regression_model.pkl')
class_model = joblib.load('model/xgb_classification_model.pkl')

# ---------------- SEND OTP ----------------
def send_otp(email, otp):
    sender_email = os.getenv("EMAIL_USER")
    sender_password = os.getenv("EMAIL_PASS")

    message = f"Your OTP is {otp}"

    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(sender_email, sender_password)
    server.sendmail(sender_email, email, message)
    server.quit()

# ---------------- REGISTER ----------------
@app.route('/', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']

        # Hash password
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

        otp = str(random.randint(100000, 999999))
        session['otp'] = otp
        session['user'] = (username, email, hashed_password.decode('utf-8'))

        send_otp(email, otp)

        return redirect('/otp')

    return render_template('register.html')

# ---------------- OTP VERIFY ----------------
@app.route('/otp', methods=['GET', 'POST'])
def otp():
    if request.method == 'POST':
        user_otp = request.form['otp']

        if user_otp == session.get('otp'):
            username, email, password = session.get('user')

            cursor.execute(
                "INSERT INTO users (username, email, password, verified) VALUES (%s,%s,%s,%s)",
                (username, email, password, True)
            )
            db.commit()

            return redirect('/login')
        else:
            return "Invalid OTP"

    return render_template('otp.html')

# ---------------- LOGIN ----------------
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        cursor.execute("SELECT password FROM users WHERE email=%s", (email,))
        user = cursor.fetchone()

        if user and bcrypt.checkpw(password.encode('utf-8'), user[0].encode('utf-8')):
            session['user'] = email
            return redirect('/dashboard')
        else:
            return "Invalid Credentials"

    return render_template('login.html')

# ---------------- DASHBOARD ----------------
@app.route('/dashboard', methods=['GET'])
def dashboard():
    if 'user' in session:
        return render_template('dashboard.html')
    return redirect('/login')

# ---------------- PREDICT ----------------
@app.route('/predict', methods=['POST'])
def predict():
    if 'user' not in session:
        return redirect('/login')

    stock = request.form.get('stock')
    if not stock:
        return redirect('/dashboard')

    try:
        # Fetch data to calculate the features the model needs
        data = yf.download(stock, period="150d")
        
        if data.empty:
            return f"Error: No data found for {stock}. Please check the ticker symbol."

        # Calculate the moving averages and target that the model uses as features
        data['MA10'] = data['Close'].rolling(window=10).mean()
        data['MA20'] = data['Close'].rolling(window=20).mean()
        data['MA30'] = data['Close'].rolling(window=30).mean()
        data['MA40'] = data['Close'].rolling(window=40).mean()
        data['MA50'] = data['Close'].rolling(window=50).mean()
        data['Target'] = data['Close'].pct_change(periods=1)

        data.dropna(inplace=True)

        if data.empty:
            return f"Error: Not enough data to calculate features for {stock}."

        # Select exact features that the model was trained on
        x = data[['MA10', 'MA20', 'MA30', 'MA40', 'MA50', 'Target']]
        latest = x.tail(1)

        # Get predictions
        future_pred_class = class_model.predict(latest)[0]
        future_pred_reg = reg_model.predict(latest)[0]

        class_result = "Up" if future_pred_class == 1 else "Down"
        # Convert regression result to float for display
        reg_result = round(float(future_pred_reg), 4)

        return render_template('result.html', stock=stock, class_result=class_result, reg_result=reg_result)

    except Exception as e:
        return f"An error occurred: {str(e)}"
# ---------------- LOGOUT ----------------
@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect('/login')

# ---------------- RUN ----------------
if __name__ == '__main__':
    app.run(debug=True)