import os
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from joblib import load, dump
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
from werkzeug.security import generate_password_hash, check_password_hash
import pymysql
from functools import wraps
import subprocess

# Flask App Setup
app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'default_secret_key')  # For better security, use an environment variable

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Database Connection
DB_USERNAME = os.getenv('DB_USERNAME', 'root')
DB_PASSWORD = os.getenv('DB_PASSWORD', 'password')
DB_HOST = os.getenv('DB_HOST', 'db')
DB_PORT = int(os.getenv('DB_PORT', 3306))
DB_NAME = os.getenv('DB_NAME', 'stock_prediction')

def get_db_connection():
    return pymysql.connect(
        host=DB_HOST,
        user=DB_USERNAME,
        password=DB_PASSWORD,
        database=DB_NAME,
        port=DB_PORT
    )

# Decorator for login-required routes
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            flash("You need to log in to access this page.", "danger")
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Train LSTM model (ensure model training is not repeated unless necessary)
def train_lstm_model(ticker, stock_data):
    model_path = os.path.join(MODEL_DIR, f"{ticker}_lstm.h5")
    scaler_path = os.path.join(MODEL_DIR, f"{ticker}_scaler.pkl")

    # Data Preprocessing
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(stock_data['Close'].values.reshape(-1, 1))

    X, y = [], []
    sequence_length = 60
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # Build and Train the LSTM Model
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=25, batch_size=32, verbose=1)
    
    model.save(model_path)
    dump(scaler, scaler_path)  # Save scaler to disk

    return model, scaler

def get_or_train_lstm_model(ticker, start_date, end_date):
    model_path = os.path.join(MODEL_DIR, f"{ticker}_lstm.h5")
    scaler_path = os.path.join(MODEL_DIR, f"{ticker}_scaler.pkl")

    stock_data = yf.download(ticker, start=start_date, end=end_date)
    if stock_data.empty:
        return None, None, {"error": "No data found for the given ticker and date range"}

    # Load pre-trained model if exists
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model = load_model(model_path)
        scaler = load(scaler_path)
        return model, scaler, None

    return train_lstm_model(ticker, stock_data)

@app.route('/')
@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = generate_password_hash(request.form['password'])

        try:
            connection = get_db_connection()
            with connection.cursor() as cursor:
                cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
                if cursor.fetchone():
                    flash("Username already exists. Please choose another.", "danger")
                    return redirect(url_for('register'))

                cursor.execute(
                    "INSERT INTO users (username, email, password) VALUES (%s, %s, %s)",
                    (username, email, password)
                )
                connection.commit()
            flash("Registration successful. Please log in.", "success")
            return redirect(url_for('login'))
        except Exception as e:
            flash(f"Error: {e}", "danger")
        finally:
            connection.close()
        return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        try:
            connection = get_db_connection()
            with connection.cursor() as cursor:
                cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
                user = cursor.fetchone()

                if user and check_password_hash(user['password'], password):
                    session['user'] = user['username']
                    flash("Login successful.", "success")
                    return redirect(url_for('home'))
                flash("Invalid username or password.", "danger")
        except Exception as e:
            flash(f"Error: {e}", "danger")
        finally:
            connection.close()
    return render_template('login.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('predict.html')

    ticker = request.form.get('ticker')
    date = request.form.get('date')

    if not ticker or not date:
        flash("Ticker and date are required.", "danger")
        return redirect(url_for('predict'))

    try:
        end_date = pd.Timestamp(date)
        start_date = end_date - pd.DateOffset(years=1)
        model, scaler, error = get_or_train_lstm_model(ticker, start_date=start_date, end_date=end_date)
        if error:
            flash(error["error"], "danger")
            return redirect(url_for('predict'))

        stock_data = yf.download(ticker, start=start_date, end=end_date)
        if stock_data.empty:
            flash("No data found for the given ticker and date range.", "danger")
            return redirect(url_for('predict'))

        last_60_days = stock_data['Close'][-60:].values.reshape(-1, 1)
        last_60_days_scaled = scaler.transform(last_60_days)
        X_test = np.array([last_60_days_scaled])
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        predicted_price = model.predict(X_test)[0][0]
        predicted_price = scaler.inverse_transform([[predicted_price]])[0][0]
        flash(f"Predicted price for {ticker} on {date} is ${round(float(predicted_price), 2)}", "success")
        return render_template('predict.html', ticker=ticker, date=date, predicted_price=predicted_price)
    except Exception as e:
        flash(f"Error: {e}", "danger")
    return redirect(url_for('predict'))

conversation_history = []

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        prompt = data.get('prompt', '').strip()

        if not prompt:
            return jsonify({"error": "Prompt cannot be empty"}), 400

        conversation_history.append(f"User: {prompt}")

        conversation = "\n".join(conversation_history)

        result = subprocess.run(
            ["ollama", "run", "llama3.2"],
            input=conversation.encode(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        if result.returncode != 0:
            error_msg = result.stderr.decode()
            return jsonify({"error": f"Ollama error: {error_msg}"}), 500

        response = result.stdout.decode().strip()
        conversation_history.append(f"Model: {response}")

        return jsonify({"response": response})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/logout')
def logout():
    session.clear()
    flash("You have been logged out.", "success")
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)