# STOCK PRICE PREDICTOR

I have trained the model with the previous its 1-year record cycle of TataGlobal share of NSE and then the model is making predictions on price of the share. For this model, I have used LSTM neural networks.

![image](https://user-images.githubusercontent.com/37563886/70072949-01029d80-161e-11ea-85f0-a2252771b4a0.png)

# INTRODUCTION TO STOCK MARKET

Broadly, stock market analysis is divided into two parts – Fundamental Analysis and Technical Analysis.
1. Fundamental Analysis involves analyzing the company’s future profitability on the basis of its current business environment and financial performance.
2. Technical Analysis, on the other hand, includes reading the charts and using statistical figures to identify the trends in the stock market.

Here I have done technical analysis of TATAGLOBAL share price of National Stock Exchange where I used Long Short Term Memory Neural Networks to make predictions on stock prices.

# LSTM NEURAL NETWORKS

Long short-term memory (LSTM) is an artificial recurrent neural network (RNN) architecture[1] used in the field of deep learning. Unlike standard feedforward neural networks, LSTM has feedback connections. It can not only process single data points (such as images), but also entire sequences of data (such as speech or video). For example, LSTM is applicable to tasks such as unsegmented, connected handwriting recognition, speech recognition and anomaly detection in network traffic or IDS's (intrusion detection systems).

A common LSTM unit is composed of a cell, an input gate, an output gate and a forget gate. The cell remembers values over arbitrary time intervals and the three gates regulate the flow of information into and out of the cell.

LSTM networks are well-suited to classifying, processing and making predictions based on time series data, since there can be lags of unknown duration between important events in a time series. LSTMs were developed to deal with the exploding and vanishing gradient problems that can be encountered when training traditional RNNs. Relative insensitivity to gap length is an advantage of LSTM over RNNs, hidden Markov models and other sequence learning methods in numerous applications.
