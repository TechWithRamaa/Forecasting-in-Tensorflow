# Forecasting-in-Tensorflow

Building forecasting systems using TensorFlow starting from generating synthetic data

Skills -> Natural Language Processing & Generating, Tokenizing, Generating n_grams, Padding, Embeddings, Bidirectional LSTM, Hyper-Parameters, Tensorflow, Keras, Python

These notebooks are submitted as part of assignments while completing a course in Coursera
* https://www.coursera.org/learn/tensorflow-sequences-time-series-and-prediction/home/info

Included notebooks are

1) https://github.com/TechWithRamaa/Forecasting-in-Tensorflow/blob/main/TimeSeries_Forecast.ipynb
    * This program generates a synthetic time series with trend, seasonality, and noise, and then explores different forecasting techniques (naive forecast, moving average, differencing)
      to predict future values, evaluating the performance using error metrics (MSE and MAE)

2) https://github.com/TechWithRamaa/Forecasting-in-Tensorflow/blob/main/TimeSeries_Forecast_With_DNN.ipynb
   * This program demonstrates the process of generating, training, and forecasting a time series using a neural network built with TensorFlow.
   * Includes the following key steps like Synthetic Time Series with a linear trend, seasonality, and noise components
   * Demonstrates the best practices for handling & organizing variables and hyperparameters by storing within a Dataclass
   * Demonstrates other key steps including Data preparation, splitting, defining architecture of DNN,
     training, Forecasting, evaluation metrics like MAE & MSE & visualization

3) https://github.com/TechWithRamaa/Forecasting-in-Tensorflow/blob/main/Forecasting_With_LSTMs.ipynb
   * This program demonstrates time series forecasting using TensorFlow with LSTM layers
   * It generates synthetic time series data, prepares it with windowing and batching, and then trains a sequential model with two LSTM layers
   * The model is optimized using the Huber loss and SGD optimizer
   * After training, the program forecasts future values and evaluates the results against the validation set
   * This approach improves the forecasting accuracy while leveraging the strengths of LSTMs for sequential data modeling

4) https://github.com/TechWithRamaa/Forecasting-in-Tensorflow/blob/main/Sunspots_Forcasting_with_A_Sophistaced_NN.ipynb
   * Building models for real world data. In particular, we will train on the Sunspots dataset: a monthly record of sunspot numbers from January 1749 to July 2018
   * This project explores various neural network architectures for time series forecasting, specifically for predicting sunspot activity
   * The architecture involves data passing through a convolutional layer, followed by stacked LSTMs, and finally stacked dense layers
   * The goal is to evaluate if this combined approach improves forecasting accuracy compared to simpler models
   * Dataset - https://www.kaggle.com/datasets/robervalt/sunspots

5) https://github.com/TechWithRamaa/Forecasting-in-Tensorflow/blob/main/Forecasting_Daily_Minimum_Temperatures_Melbourne.ipynb
   * We will be using the Daily Minimum Temperatures in Melbourne dataset which contains data of the daily minimum temperatures recorded in Melbourne from 1981 to 1990
   * This project explores various neural network architectures for time series forecasting & experimenting various optimizers & loss functions to improve the performance of the model
   * The architecture involves data passing through a convolutional layer, followed by stacked LSTMs, and finally stacked dense layers
   * The goal is to evaluate if this combined approach improves forecasting accuracy compared to simpler models
   * Dataset - https://github.com/jbrownlee/Datasets/blob/master/daily-min-temperatures.csv

