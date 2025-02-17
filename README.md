# License-Plate-Recognition
License Plate Recognition For Car With Python And OpenCV

### 算法实现：
* 算法思想来自于网上资源，先使用图像边缘和车牌颜色定位车牌，根据直方图分割字符，最后拟用SVM再分类识别字符。  
* 数据集来自 https://github.com/liuruoze/EasyPR/tree/master/resources/train, 并且自己做了部分数据增强。

##### 额外说明：测试中发现，车牌定位算法的参数受图像分辨率、色偏、车距影响(其他图片很可能因为像素等问题识别不了，识别其他像素的车牌需要修改config文件里面的参数).
##### SVM说明：如需要重新训练，请把原来的svm.dat和svmchinese.dat删除再运行predict_hang.py.
## 针对于大作业

- ### 基于以下假设

1. 对于不同车距，车牌的大小不一，所以这里统一处理的时候针对medium和difficult有着不同的resize大小(medium的车牌比difficult车牌面积大)；
2. 对于不同车距，车牌的清晰度不一，所以这里检测边缘的时候对于模糊边缘(difficult)只检测一个方向，对于较清晰边缘(medium)检测了两个方向, 并且对模糊车牌(difficult)的长宽比要求放宽；
3. 在HSV空间中，对于同色度的车牌与背景，车牌的饱和度与明度均比背景高(颜色定位中滤除非车牌矩形时S、V的阈值区分)；
4. 对于绿色车牌，相比于蓝色车牌，认为绿色车牌存在渐变且饱和度较低(颜色定位中缩进车牌区域时S、V的阈值与缩进条件区分)；

- [x] de一下颜色定位的bug，目前在difficult的几张图片的hsv映射有问题；
- [x] 对定位好的车牌上的字符进行分割；
- [x] 对分割好的车牌字符块搭建svm模型训练识别车牌上的字符或数字；
- [x] 优化识别准确度，目前的想法是从数据集下手(针对测试集中展现的识别效果不好的个例)，已测，效果很好！

### Done！开始写报告吧！
