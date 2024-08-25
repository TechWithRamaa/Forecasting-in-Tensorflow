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


https://github.com/TechWithRamaa/NLP-in-Tensorflow/blob/main/Creating_Poetry_With_Bidirectional_LSTMs.ipynb

Developed a creative tool by building a poetry generator
Drawing inspiration from traditional Irish songs and Shakespearean poetry, trained our model to generate beautiful verses
Tokenizing, Generating n_grams, Padding, defined model with Embedding layer & Bidirectional LSTM, Generated verses with 100 words by inputting seed texts
Dataset - https://www.opensourceshakespeare.org/views/sonnets/sonnet_view.php?range=viewrange&sonnetrange1=1&sonnetrange2=154

