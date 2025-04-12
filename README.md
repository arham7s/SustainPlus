# SustainPlus: Stock Price Prediction with ESG Integration

## Live Demo
Access the live application at: [https://sutainplus.streamlit.app/](https://sutainplus.streamlit.app/)

## Overview
SutainPlus is a production-grade Streamlit application designed to forecast future stock prices using deep learning techniques and rank stocks based on a combination of predicted financial performance and ESG (Environmental, Social, and Governance) scores. The app downloads historical stock data from yFinance and employs a GRU-based model with dropout regularization and early stopping to ensure robust predictions. By combining normalized predicted profit with normalized ESG scores—according to user-defined weights—the app generates an integrated ranking that highlights both growth potential and sustainability.

## Features
- **Stock Price Forecasting:**  
  Uses a GRU-based deep learning model to predict future stock prices from historical data.
- **ESG Integration:**  
  Combines ESG scores with predicted profit to derive balanced stock rankings.
- **Interactive Dashboard:**  
  Allows users to input stock tickers, select the prediction period, and adjust weighting between financial and ESG metrics.
- **Visual Analytics:**  
  Displays historical versus predicted price charts and key accuracy metrics such as RMSE and MAE.

## Technologies Used
- **Language:** Python 3.x  
- **Web Framework:** Streamlit  
- **Deep Learning:** TensorFlow/Keras (GRU, Dropout, Early Stopping)  
- **Data Acquisition:** yFinance  
- **Data Processing:** Pandas, NumPy, Scikit-Learn  
- **Visualization:** Matplotlib


