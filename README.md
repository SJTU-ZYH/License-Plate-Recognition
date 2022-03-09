# License-Plate-Recognition
License Plate Recognition For Car With Python And OpenCV

#### 用python3+opencv3做的中国车牌识别，包括算法和客户端界面，只有2个文件，surface.py是界面代码，predict.py是算法代码，界面不是重点所以用tkinter写得很简单。

### 使用方法：
版本：python3.4.4，opencv3.4和numpy1.14和PIL5<br>
下载源码，并安装python、numpy、opencv的python版、PIL，运行surface.py即可

### 算法实现：
算法思想来自于网上资源，先使用图像边缘和车牌颜色定位车牌，再识别字符。车牌定位在predict方法中，为说明清楚，完成代码和测试后，加了很多注释，请参看源码。车牌字符识别也在predict方法中，请参看源码中的注释，需要说明的是，车牌字符识别使用的算法是opencv的SVM， opencv的SVM使用代码来自于opencv附带的sample，StatModel类和SVM类都是sample中的代码。SVM训练使用的训练样本来自于github上的EasyPR的c++版本。由于训练样本有限，你测试时会发现，车牌字符识别，可能存在误差，尤其是第一个中文字符出现的误差概率较大。源码中，我上传了EasyPR中的训练样本，在train\目录下，如果要重新训练请解压在当前目录下，并删除原始训练数据文件svm.dat和svmchinese.dat。

##### 额外说明：算法代码只有500行，测试中发现，车牌定位算法的参数受图像分辨率、色偏、车距影响（test目录下的车牌的像素都比较小，其他图片很可能因为像素等问题识别不了，识别其他像素的车牌需要修改config文件里面的参数，此项目仅是抛砖引玉，提供一个思路）。
## 针对于大作业

- ### 基于以下假设

1. 对于不同车距，车牌的大小不一，所以这里统一处理的时候针对medium和difficult有着不同的resize大小(medium的车牌比difficult车牌面积大)；
2. 对于不同车距，车牌的清晰度不一，所以这里检测边缘的时候对于模糊边缘(difficult)只检测一个方向，对于较清晰边缘(medium)检测了两个方向；
3. 在HSV空间中，对于同色度的车牌与背景，车牌的饱和度与明度均比背景高；
4. 

- [ ] de一下颜色定位的bug，目前在difficult的几张图片的hsv映射有问题；
- [ ] 对定位好的车牌进行resize后训练识别车牌上的字符、数字的网络模型；
