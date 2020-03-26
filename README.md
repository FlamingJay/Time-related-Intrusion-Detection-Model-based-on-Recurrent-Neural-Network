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

If the method does help for you in your paper, please cite:

Lin Y, Wang J, Tu Y, et al. Time-Related Network Intrusion Detection Model: A Deep Learning Method[C]//2019 IEEE Global Communications Conference (GLOBECOM). IEEE, 2019: 1-6.

Likewise, if there is any question, contact me heuwangjie@hrbeu.edu.cn. Thanks.


本实验主要完成以下内容：
1.构建一个稀疏自编码器，完成降维任务
2.构建LSTM（GRU、双向LSTM、双向GRU），完成数据特征挖掘
3.最后以一个sigmoid神经元完成二分类任务，以binary_crossentropy作为衡量指标

预期：
1.不同DAE结构，对于分类的影响。
    这个过程实际为pre-training过程，以DAE+单步LSTM（GRU及双向）结构完成任务
    最终确定所需参数，包括DAE神经元个数以及稀疏系数rho的确定，需要给出一个表table
2.不同LSTM、GRU及双向，共4个的对比
    这个过程实际为fine-tuning过程，以不同步数+结构完成任务
    最终确定最优的结构，包括选用的cell是什么以及步数的确定，需要给出一个图figure（4条曲线，在不同步数的准确率和虚警率对比）
3.选出最优结构，然后进行test一维图的显示，figure
