# 本程序是用于绘制出测试数据的判决情况，并采用一个相似度来衡量
# 参考网址：https://matplotlib.org/gallery/recipes/transparent_legends.html#sphx-glr-gallery-recipes-transparent-legends-py
# https://matplotlib.org/gallery/lines_bars_and_markers/cohere.html#sphx-glr-gallery-lines-bars-and-markers-cohere-py
# 用蓝色表示原始的数据，用橙色表示预测的数据，其中预测若出现失误，则对应点采用的是红色的，即共有三种颜色。
# 第二幅图画出的是预测错误的累计个数

import numpy as np
import matplotlib.pyplot as plt


# Two signals with a coherent part at 10Hz and a random part
s1 = np.load('data/test_label.npy')
s2 = np.load('data/plot_prediction.npy')
test_pred = s2 > 0.5
index_same = np.argwhere(s1 == test_pred)
index_diff = np.argwhere(s1 != test_pred)

print(index_same.shape)
print(index_diff.shape)
dt = 1.0
t = np.arange(0, len(s1), dt)
s3 = np.ones(len(s1)) * 0.5
fig = plt.figure(1)
ax1 = fig.add_subplot(111)

font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 14,
}
font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 14,
}
ax1.plot(t, s1, markersize=8, label='True Label', color='blue', marker='o', linestyle='-')
ax1.plot(t[index_same[:,0]], s2[index_same[:,0]], markersize=2, label='Correct Prediction', color='orange', marker='s', linestyle='')
ax1.plot(t[index_diff[:,0]], s2[index_diff[:,0]], markersize=0.5, label='Wrong Prediction', color='red', marker='s', linestyle='')
ax1.plot(t, s3, 'r-')
ax1.set_ylim(-0.3, 1.5)
ax1.set_xlabel('samples', font2)
ax1.set_ylabel('Probability of Each Sample', font2)
ax1.legend(loc='upper right', prop=font2)
plt.title('Comparsion of Prediction and True Labels', font1)
ax1.grid(True)
plt.show()
# 画出累计
fig2 = plt.figure(1)
ax2 = fig2.add_subplot(111)
count_line = np.zeros(len(s1))
index_low = 0
index_high = 0
for i, index in enumerate(index_diff):

    index_high = index[0]
    count_line[index_low:index_high] = i
    index_low = index_high
count_line[81918:] = 1506
ax2.plot(t, count_line)
plt.title('Cumulative Amount of Incorrect Predictions', font1)
ax2.set_xlabel('samples', font2)
ax2.set_ylabel('Number of Incorrect Predictions', font2)

# plt.subplots_adjust(wspace=0., hspace =0.3)
plt.show()
