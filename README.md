# Time-related-Intrusion-Detection-Model-based-on-Recurrent-Neural-Network
Here, we use RNN to deal with the network intrusion problem. The UNSW-NB15 dataset is used.

Totally, we divide the process into two parts.
![image](https://github.com/FlamingJay/Time-related-Intrusion-Detection-Model-based-on-Recurrent-Neural-Network/blob/master/figure/framework.png)

The first part is regarded as pre-training. The stacked sparse autoencoder is used for feature extraction and dimension reduction.
The structure of SSAE is as follow:
![image](https://github.com/FlamingJay/Time-related-Intrusion-Detection-Model-based-on-Recurrent-Neural-Network/blob/master/figure/Sparse%20AE.png)

After that, the data can be in low-dimension.

Then, we organize the 2D traffic into 3D data, that is, put a few samples together as a time-seires sample.

Finally, different variant RNNs are adopted to classify the current data into normal or anomaly.
The structure of LSTM is:
![image](https://github.com/FlamingJay/Time-related-Intrusion-Detection-Model-based-on-Recurrent-Neural-Network/blob/master/figure/LSTM.png)

The structure of GRU is:
![image](https://github.com/FlamingJay/Time-related-Intrusion-Detection-Model-based-on-Recurrent-Neural-Network/blob/master/figure/GRU.png)

The final result of classification can be shown in this way:
![image](https://github.com/FlamingJay/Time-related-Intrusion-Detection-Model-based-on-Recurrent-Neural-Network/blob/master/figure/wave_1.png)
