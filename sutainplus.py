# -*- coding: utf-8 -*-
"""
SutainPlus Streamlit App
A complete application that predicts stock prices with ESG integration.
If the ESG data CSV file is not found, dummy data will be used.
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Check if the ESG CSV file exists. If not, use dummy ESG data.
esg_file_path = 'esg_data.csv'
if os.path.exists(esg_file_path):
    esg_data = pd.read_csv(esg_file_path)
    st.info(f"Loaded ESG data from: {esg_file_path}")
else:
    st.warning(f"CSV file '{esg_file_path}' not found. Using dummy ESG data.")
    esg_data = pd.DataFrame({
        'ticker': ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META'],
        'total_score': [80, 85, 75, 90, 70]
    })

def create_dataset(dataset, time_step=1):
    """Convert time series into samples for training/testing."""
    X, Y = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]
        X.append(a)
        Y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(Y)

def predict_future(model, data, days_to_predict):
    """Predict future stock prices given the last 100 days data."""
    last_100_days = data[-100:].reshape(1, -1, 1)
    future_predictions = []
    for _ in range(days_to_predict):
        next_day_prediction = model.predict(last_100_days)
        future_predictions.append(next_day_prediction[0, 0])
        # Roll and update the input array with the new prediction.
        last_100_days = np.roll(last_100_days, -1, axis=1)
        last_100_days[0, -1, 0] = next_day_prediction
    return np.array(future_predictions).reshape(-1, 1)

def load_model_and_data(tickers, esg_data):
    """Download stock data, create/train LSTM models, and calculate metrics."""
    data = {}
    scaled_data = {}
    scalers = {}
    models = {}
    accuracies = {}

    # Convert ESG data from dataframe to dictionary
    esg_scores = esg_data.set_index('ticker')['total_score'].to_dict()

    for ticker in tickers:
        try:
            stock_data = yf.download(ticker, start="2010-01-01", end="2023-12-31", progress=False)
            if not stock_data.empty:
                df = stock_data[['Close']]
                dataset = df.values
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaled = scaler.fit_transform(dataset)
                data[ticker] = dataset
                scaled_data[ticker] = scaled
                scalers[ticker] = scaler

                # Split data into training and testing sets
                all_scaled_data = scaled_data[ticker]
                training_size = int(len(all_scaled_data) * 0.65)
                train_data = all_scaled_data[:training_size, :]
                test_data = all_scaled_data[training_size:, :]

                time_step = 100
                X_train, y_train = create_dataset(train_data, time_step)
                X_test, y_test = create_dataset(test_data, time_step)
                X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
                X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

                # Build and train the LSTM model
                model = Sequential()
                model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
                model.add(LSTM(50, return_sequences=True))
                model.add(LSTM(50))
                model.add(Dense(1))
                model.compile(loss='mean_squared_error', optimizer='adam')

                # Use a small number of epochs for faster processing during testing
                model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1, batch_size=128, verbose=0)
                models[ticker] = model

                # Calculate accuracy metrics
                y_pred = model.predict(X_test)
                y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
                y_pred_inv = scaler.inverse_transform(y_pred)

                rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
                mae = mean_absolute_error(y_test_inv, y_pred_inv)

                accuracies[ticker] = {'RMSE': rmse, 'MAE': mae}

            else:
                st.warning(f"No data available for {ticker}")
        except Exception as e:
            st.error(f"Error processing {ticker}: {str(e)}")
    return data, scaled_data, scalers, models, esg_scores, accuracies

def show_stock_price_prediction_page():
    """Main page function for Streamlit app."""
    st.title("Stock Price Prediction with ESG Integration")

    # User inputs for tickers and prediction period
    tickers_input = st.text_input("Enter stock tickers (comma separated):", "AAPL, MSFT, AMZN, GOOGL, META")
    tickers = [ticker.strip().upper() for ticker in tickers_input.split(",") if ticker.strip() != ""]
    
    period = st.selectbox("Select prediction period:", ("3 months", "6 months", "1 year"))
    days_to_predict = {"3 months": 90, "6 months": 180, "1 year": 365}[period]

    # Weights for the ranking based on price predictions and ESG score
    price_weight = st.slider("Price-ESG weight:", 0.0, 1.0, 0.5)
    esg_weight = 1 - price_weight

    if st.button("Get Predictions"):
        if not tickers:
            st.error("Please enter at least one ticker.")
            return

        # Load data, models, and ESG scores
        data, scaled_data, scalers, models, esg_scores, accuracies = load_model_and_data(tickers, esg_data)
        if not models:
            st.error("Failed to load models or data for the provided tickers.")
            return

        predictions_dict = {}
        for ticker in tickers:
            if ticker in scaled_data:
                # Generate future predictions using the trained model
                future_predictions = predict_future(models[ticker], scaled_data[ticker], days_to_predict)
                actual_predictions = scalers[ticker].inverse_transform(future_predictions)
                predictions_dict[ticker] = actual_predictions

                st.subheader(f"Predicted stock prices for {ticker} for the next {period}:")
                predictions_df = pd.DataFrame({
                    "Day": range(1, len(actual_predictions) + 1),
                    "Predicted Price": actual_predictions.flatten()
                })
                st.dataframe(predictions_df)

                # Plotting actual and predicted prices
                plt.figure(figsize=(10, 5))
                plt.plot(data[ticker], label=f"{ticker} Actual")
                plt.plot(range(len(data[ticker]), len(data[ticker]) + days_to_predict), 
                         actual_predictions, label=f"{ticker} Predicted ({period})")
                plt.xlabel("Days")
                plt.ylabel("Price")
                plt.title(f"Stock Price Predictions for {ticker}")
                plt.legend()
                st.pyplot(plt.gcf())
                plt.close()

                # Display accuracy metrics
                if ticker in accuracies:
                    st.write(f"Accuracy metrics for {ticker}:")
                    st.write(f"RMSE: {accuracies[ticker]['RMSE']:.2f}")
                    st.write(f"MAE: {accuracies[ticker]['MAE']:.2f}")
            else:
                st.warning(f"Skipping {ticker}, as no scaled data is available.")

        # Ranking stocks based on predicted profit and ESG score
        profit_ranks = {ticker: np.mean(predictions_dict[ticker]) for ticker in tickers if ticker in predictions_dict}
        if profit_ranks:
            normalized_profit_ranks = MinMaxScaler().fit_transform(np.array(list(profit_ranks.values())).reshape(-1, 1)).flatten()
            normalized_esg_scores = MinMaxScaler().fit_transform(np.array([esg_scores.get(ticker, 0) for ticker in tickers]).reshape(-1, 1)).flatten()
            combined_ranks = {ticker: (normalized_profit_ranks[i] * price_weight + normalized_esg_scores[i] * esg_weight)
                              for i, ticker in enumerate(tickers)}
            ranked_stocks = sorted(combined_ranks.items(), key=lambda x: x[1], reverse=True)

            st.subheader("Ranked Stocks based on predicted profit and ESG score:")
            for rank, (ticker, score) in enumerate(ranked_stocks, 1):
                st.write(f"{rank}. {ticker}: {score:.2f}")
        else:
            st.write("No predictions available to rank stocks.")

# Run the Streamlit app
if __name__ == '__main__':
    show_stock_price_prediction_page()

