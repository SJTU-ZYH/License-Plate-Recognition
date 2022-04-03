import time

import cv2
import numpy as np
from numpy.linalg import norm
import sys
import os
import json

# for difficult
MAX_WIDTH_DIF = 781
# for medium
MAX_WIDTH_MED = 450
# for easy
MAX_WIDTH_EAS = 200

SZ = 20          #训练图片长宽
bin_n = 16       #训练时的图片分块数
Min_Area = 1000  #车牌区域允许最小面积
PROVINCE_START = 1000

svm_params = dict(kernel_type=cv2.ml.SVM_LINEAR,
                  svm_type=cv2.ml.SVM_C_SVC,
                  C=6, gamma=10) # C=3, gamma=8


def imreadex(filename):
    return cv2.imdecode(np.fromfile(filename, dtype=np.uint8), cv2.IMREAD_COLOR)


def point_limit(point):
    if point[0] < 0:
        point[0] = 0
    if point[1] < 0:
        point[1] = 0


#根据设定的阈值和图片直方图，找出波峰，用于分隔字符
def find_waves(threshold, histogram):
    # print("histogram size: " + str(histogram.size))
    up_point = -1#上升点
    is_peak = False
    if histogram[0] > threshold:
        # print("yes")
        up_point = 0
        is_peak = True
    wave_peaks = []
    for i,x in enumerate(histogram):
        # if up_point >= 1:
        #     print("i, x, threshold:", i, x, threshold)
        if is_peak and x < threshold:
            if i - up_point > 4:
                is_peak = False
                wave_peaks.append((up_point, i))
        elif not is_peak and x >= threshold:
            is_peak = True
            up_point = i
    if is_peak and up_point != -1 and i - up_point > 4:
        # print("yes")
        wave_peaks.append((up_point, i))
    # print("wave_peaks:", wave_peaks)
    return wave_peaks


#根据找出的波峰，分隔图片，从而得到逐个字符图片
def seperate_card(img, waves):
    part_cards = []
    for wave in waves:
        part_cards.append(img[:, wave[0]:wave[1]])
    return part_cards


# 来自opencv的sample，用于svm训练
def deskew(img):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11'] / m['mu02']
    M = np.float32([[1, skew, -0.5 * SZ * skew], [0, 1, 0]])
    img = cv2.warpAffine(img, M, (SZ, SZ), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img


# 来自opencv的sample，用于svm训练
def preprocess_hog(digits):
    samples = []
    for img in digits:
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
        mag, ang = cv2.cartToPolar(gx, gy)
        bin_n = 16
        bin = np.int32(bin_n * ang / (2 * np.pi))
        # print(bin.shape)
        bin_cells = bin[:10, :10], bin[10:, :10], bin[:10, 10:], bin[10:, 10:]
        mag_cells = mag[:10, :10], mag[10:, :10], mag[:10, 10:], mag[10:, 10:]
        hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
        hist = np.hstack(hists)

        # transform to Hellinger kernel
        eps = 1e-7
        hist /= hist.sum() + eps
        hist = np.sqrt(hist)
        hist /= norm(hist) + eps
        # print("hist:", type(hist), '\n', hist.shape, hist)
        samples.append(hist)
    return np.float32(samples)

provinces = [
            "zh_hu", "沪",
            "zh_lu", "鲁",
            "zh_wan", "皖",
            "zh_yu", "豫"
        ]


class StatModel(object):
    def load(self, fn):
        self.model = self.model.load(fn)
    def save(self, fn):
        self.model.save(fn)


class SVM(StatModel):
    def __init__(self, C = 1, gamma = 0.5):
        self.model = cv2.ml.SVM_create()
        self.model.setGamma(gamma)
        self.model.setC(C)
        self.model.setKernel(cv2.ml.SVM_RBF)
        self.model.setType(cv2.ml.SVM_C_SVC)
    #训练svm
    def train(self, samples, responses):
        self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)
    #字符识别
    def predict(self, samples):
        r = self.model.predict(samples)
        return r[1].ravel()


class CardPredictor:
    # type=0, 中等
    # type=1, 困难
    def __init__(self, type = 0):
        self.type = type
        if self.type == 1:
            self.MAX_WIDTH = MAX_WIDTH_DIF
            self.MIN_RATIO = 1.5
        elif self.type == 0:
            self.MAX_WIDTH = MAX_WIDTH_MED
            self.MIN_RATIO = 2.5
        else:
            self.MAX_WIDTH = MAX_WIDTH_EAS
            self.MIN_RATIO = 2
        self.MAX_RATIO = 7.5
        # 车牌识别的部分参数保存在js中，便于根据图片分辨率做调整
        f = open('config.json')
        j = json.load(f)
        for c in j["config"]:
            if c["open"]:
                self.cfg = c.copy()
                break
        else:
            raise RuntimeError('没有设置有效配置参数')
        # 车牌定位结果
        self.card_imgs = []
        self.card_imgs_path = []
        self.colors = []
        # 车牌字符分割结果
        self.card_num = ""
        self.part_cards = []
        self.predict_result = []

    def __del__(self):
        self.save_traindata()

    def train_svm(self):
        # 识别英文字母和数字
        self.model = SVM(C=svm_params['C'], gamma=svm_params['gamma'])
        # 识别中文
        self.modelchinese = SVM(C=svm_params['C'], gamma=svm_params['gamma'])
        if os.path.exists("svm.dat"):
            self.model.load("svm.dat")
        else:
            chars_train = []
            chars_label = []

            for root, dirs, files in os.walk("traindata\\chars2"):
                if len(os.path.basename(root)) > 1:
                    continue
                root_int = ord(os.path.basename(root))
                for filename in files:
                    filepath = os.path.join(root, filename)
                    digit_img = cv2.imread(filepath)
                    digit_img = cv2.cvtColor(digit_img, cv2.COLOR_BGR2GRAY)
                    chars_train.append(digit_img)
                    # chars_label.append(1)
                    chars_label.append(root_int)

            chars_train = list(map(deskew, chars_train))
            chars_train = preprocess_hog(chars_train)
            # print(chars_train.shape)
            # chars_train = chars_train.reshape(-1, 20, 20).astype(np.float32)
            chars_label = np.array(chars_label)
            self.model.train(chars_train, chars_label)
        if os.path.exists("svmchinese.dat"):
            self.modelchinese.load("svmchinese.dat")
        else:
            chars_train = []
            chars_label = []
            for root, dirs, files in os.walk("traindata\\charsChinese"):
                if not os.path.basename(root).startswith("zh_"):
                    continue
                pinyin = os.path.basename(root)
                index = provinces.index(pinyin) + PROVINCE_START + 1  # 1是拼音对应的汉字
                for filename in files:
                    filepath = os.path.join(root, filename)
                    digit_img = cv2.imread(filepath)
                    digit_img = cv2.cvtColor(digit_img, cv2.COLOR_BGR2GRAY)
                    chars_train.append(digit_img)
                    # chars_label.append(1)
                    chars_label.append(index)
            chars_train = list(map(deskew, chars_train))
            chars_train = preprocess_hog(chars_train)
            # chars_train = chars_train.reshape(-1, 20, 20).astype(np.float32)
            chars_label = np.array(chars_label)
            # print(chars_train.shape)
            self.modelchinese.train(chars_train, chars_label)

    def save_traindata(self):
        if not os.path.exists("svm.dat"):
            self.model.save("svm.dat")
        if not os.path.exists("svmchinese.dat"):
            self.modelchinese.save("svmchinese.dat")


    def accurate_place(self, card_img_hsv, Hlimit1, Hlimit2, Smin, color):
        row_num, col_num = card_img_hsv.shape[:2]
        # print("row, col num:", row_num, col_num)
        xl = col_num
        xr = 0
        yh = 0
        yl = row_num
        # col_num_limit = self.cfg["col_num_limit"]
        Smin = Smin * 0.35 if color == "green" else Smin # 绿色车牌较浅
        row_num_limit = self.cfg["row_num_limit"] if color == "green" else 0.75 * row_num # 绿色有渐变
        col_num_limit = col_num * 0.2 if color != "green" else col_num * 0.3  # 绿色有渐变
        for i in range(row_num):
            count = 0
            for j in range(col_num):
                H = card_img_hsv.item(i, j, 0)
                S = card_img_hsv.item(i, j, 1)
                V = card_img_hsv.item(i, j, 2)
                if Hlimit1 < H <= Hlimit2 and S > Smin and 60 < V:
                    count += 1
            # print('accurate count for y:', count)
            # print('i:',i,'count:',count)
            if count > col_num_limit:
                # print("yeah!")
                if yl > i:
                    yl = i
                if yh < i:
                    yh = i
                # print('yl, yh:', yl, '\t', yh)
        for j in range(col_num):
            count = 0
            for i in range(row_num):
                H = card_img_hsv.item(i, j, 0)
                S = card_img_hsv.item(i, j, 1)
                V = card_img_hsv.item(i, j, 2)
                if Hlimit1 < H <= Hlimit2 and S > Smin and 60 < V:
                    count += 1
            # print('j:',j,'count:',count)
            if count > row_num - row_num_limit:
            # if count > 0.25 * row_num:
                if xl > j:
                    xl = j
                if xr < j:
                    xr = j
        return xl, xr, yh+2, yl-2

    def locate(self, car_pic, resize_rate=1):
        self.card_imgs = []
        self.colors = []
        t0 = time.time()
        print('---------------------------------------------')
        print('对图片', car_pic, '的车牌定位')
        if type(car_pic) == type(""):
            img = imreadex(car_pic)
        else:
            img = car_pic
        # print('time_use_read:', time.time() - t0)
        if self.type == 1:
            img = img[int(img.shape[0] * 2 / 5): img.shape[0]]
            pic_hight, pic_width = img.shape[:2]
        else:
            pic_hight, pic_width = img.shape[:2]
        if pic_width > self.MAX_WIDTH:
            pic_rate = self.MAX_WIDTH / pic_width
            img = cv2.resize(img, (self.MAX_WIDTH, int(pic_hight * pic_rate)), interpolation=cv2.INTER_LANCZOS4)
            pic_hight, pic_width = img.shape[:2]

        if resize_rate != 1:
            img = cv2.resize(img, (int(pic_width * resize_rate), int(pic_hight * resize_rate)),
                             interpolation=cv2.INTER_LANCZOS4)
            pic_hight, pic_width = img.shape[:2]
        # cv2.imshow('original', img)
        # print("h,w:", pic_hight, pic_width)
        # pic_hight, pic_width = img.shape[:2]
        # print("h,w:", img.shape)
        if self.type == 2:
            self.card_imgs.append(img)
        else:
            blur = self.cfg["blur"]
            oldimg = img

            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # cv2.imshow('gray', img)
            # 高斯去噪
            if blur > 0:
                img = cv2.GaussianBlur(img, (blur, blur), 0)  # 图片分辨率调整
            # 中值滤波
            median = cv2.medianBlur(img, 5)

            # cv2.imshow('gray', img)
            # cv2.imshow('median', median)
            # 去掉图像中不会是车牌的区域
            kernel = np.ones((20, 20), np.uint8)
            img_opening = cv2.morphologyEx(median, cv2.MORPH_OPEN, kernel)
            # cv2.imshow('addweight_before', img_opening)
            img_opening = cv2.addWeighted(median, 1, img_opening, -1, 0)
            # cv2.imshow('addweight', img_opening)
            # for difficult
            if self.type:
                sobel = cv2.Sobel(img_opening, cv2.CV_8U, 1, 0, ksize=3)
            # for medium
            else:
                sobel = cv2.Sobel(img_opening, cv2.CV_8U, 1, 1, ksize=3)
            # 二值化
            ret, binary = cv2.threshold(sobel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # cv2.imshow('thresh', binary)
            # img_edge = cv2.Canny(img_thresh, 100, 200)
            # cv2.imshow('edge', img_edge)
            # 使用开运算和闭运算让图像边缘成为一个整体
            kernel = np.ones((self.cfg["morphologyr"], self.cfg["morphologyc"]), np.uint8)
            img_edge1 = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            # cv2.imshow('edge1', img_edge1)
            img_edge2 = cv2.morphologyEx(img_edge1, cv2.MORPH_OPEN, kernel)
            # cv2.imshow('edge2', img_edge2)

            # 查找图像边缘整体形成的矩形区域，可能有很多，车牌就在其中一个矩形区域中
            try:
                contours, hierarchy = cv2.findContours(img_edge2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            except ValueError:
                image, contours, hierarchy = cv2.findContours(img_edge2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours = [cnt for cnt in contours if cv2.contourArea(cnt) > Min_Area]
            print('len(contours)', len(contours))
            # print(contours)
            # cv2.drawContours(oldimg, contours, 0, (0, 0, 255), 2)
            # cv2.imshow('contours', oldimg)

            # 一一排除不是车牌的矩形区域
            car_contours = []
            old_img = oldimg.copy()
            for cnt in contours:
                rect = cv2.minAreaRect(cnt)
                area_width, area_height = rect[1]
                if area_width < area_height:
                    area_width, area_height = area_height, area_width
                wh_ratio = area_width / area_height
                # print(wh_ratio)
                # 要求矩形区域长宽比在2到5.5之间，2到5.5是车牌的长宽比，其余的矩形排除
                if wh_ratio > self.MIN_RATIO and wh_ratio < self.MAX_RATIO:
                    car_contours.append(rect)
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    old_img = cv2.drawContours(old_img, [box], 0, (0, 0, 255), 2)
            # cv2.imshow("edge4", old_img)
                # cv2.waitKey(0)

            # print(len(car_contours))

            print("精确定位")
            # self.card_imgs = []
            # 矩形区域可能是倾斜的矩形，需要矫正，以便使用颜色定位
            s = 0
            # print("car_contours:", car_contours)
            for rect in car_contours:
                if rect[2] > -1 and rect[2] < 1:  # 创造角度，使得左、高、右、低拿到正确的值
                    angle = 1
                else:
                    angle = rect[2]
                rect = (rect[0], (rect[1][0] + 5, rect[1][1] + 5), angle)  # 扩大范围，避免车牌边缘被排除
                # print("rect", rect)

                box = cv2.boxPoints(rect)
                # print("box:", box)
                heigth_point = right_point = [0, 0]
                left_point = low_point = [pic_width, pic_hight]
                # print(left_point, low_point, right_point, heigth_point)
                for point in box:
                    if left_point[0] > point[0]:
                        left_point = point
                    if low_point[1] > point[1]:
                        low_point = point
                    if heigth_point[1] < point[1]:
                        heigth_point = point
                    if right_point[0] < point[0]:
                        right_point = point
                # print(left_point, low_point, right_point, heigth_point)
                # cv2.imshow('oldimg_', oldimg)
                if left_point[1] < right_point[1]:  # 正角度
                    # print("正/无角度")
                    new_right_point = [right_point[0], heigth_point[1]]
                    pts2 = np.float32([left_point, heigth_point, new_right_point])  # 字符只是高度需要改变
                    pts1 = np.float32([left_point, heigth_point, right_point])
                    # print("pts1:", pts1, '\npts2:', pts2)
                    M = cv2.getAffineTransform(pts1, pts2)
                    dst = cv2.warpAffine(oldimg, M, (pic_width, pic_hight))
                    # cv2.imshow('dst+'+str(s), dst)
                    point_limit(new_right_point)
                    point_limit(heigth_point)
                    point_limit(left_point)

                    card_img = dst[int(left_point[1]):int(heigth_point[1]), int(left_point[0]):int(new_right_point[0])]
                    self.card_imgs.append(card_img)
                # cv2.imshow("card", card_img)
                # cv2.waitKey(0)
                elif left_point[1] > right_point[1]:  # 负角度
                    # print("负角度")
                    new_left_point = [left_point[0], heigth_point[1]]
                    pts2 = np.float32([new_left_point, heigth_point, right_point])  # 字符只是高度需要改变
                    pts1 = np.float32([left_point, heigth_point, right_point])
                    # print("pts1:", pts1, '\npts2:', pts2)
                    M = cv2.getAffineTransform(pts1, pts2)
                    dst = cv2.warpAffine(oldimg, M, (pic_width, pic_hight))
                    # cv2.imshow('dst-'+str(s), dst)
                    point_limit(right_point)
                    point_limit(heigth_point)
                    point_limit(new_left_point)
                    card_img = dst[int(right_point[1]):int(heigth_point[1]), int(new_left_point[0]):int(right_point[0])]
                    self.card_imgs.append(card_img)
                else:
                    card_img_ = oldimg[int(left_point[1]):int(heigth_point[1]), int(left_point[0]):int(right_point[0])]
                    # cv2.imshow("card"+'haha', card_img)
                    # cv2.waitKey(0)
                    self.card_imgs.append(card_img_)
                # cv2.imshow("card"+str(s), card_img)
                # cv2.waitKey(0)
                s+=1
        # 开始使用颜色定位，排除不是车牌的矩形，目前只识别蓝、绿、黄车牌
        # self.colors = []
        t = 0
        for card_index, card_img in enumerate(self.card_imgs):
            green = yello = blue = black = white = 0
            # print(card_img)
            # cv2.imshow('card_img' + str(8), card_img)
            # cv2.waitKey(0)
            try:
                card_img_hsv = cv2.cvtColor(card_img, cv2.COLOR_BGR2HSV)
            except:
                card_img_hsv = None
            # 有转换失败的可能，原因来自于上面矫正矩形出错
            if card_img_hsv is None:
                continue
            # card_img = cv2.cvtColor(card_img, cv2.COLOR_BGR2GRAY)
            # card_img = cv2.cvtColor(card_img, cv2.COLOR_GRAY2BGR)
            # card_img_hsv = cv2.cvtColor(card_img, cv2.COLOR_BGR2HSV)
            # cv2.imshow('card_img'+str(t), card_img)
            # cv2.imshow('card_img_hsv'+str(t), card_img_hsv)
            # cv2.imwrite('card_img' + str(t) + '.jpg', card_img)
            # cv2.imwrite('card_img_hsv'+str(t)+'.jpg', card_img_hsv)
            t+=1

            row_num, col_num = card_img_hsv.shape[:2]
            card_img_count = row_num * col_num
            for i in range(row_num):
                for j in range(col_num):
                    H = card_img_hsv.item(i, j, 0)
                    S = card_img_hsv.item(i, j, 1)
                    V = card_img_hsv.item(i, j, 2)
                    if 11 < H <= 34 and S > 70 and V >= 75:  # 图片分辨率调整
                        yello += 1
                    elif 35 < H <= 99 and S > 34 and V >= 75:  # 图片分辨率调整
                        green += 1
                    elif 99 < H <= 124 and S > 60 and V >= 75:  # 图片分辨率调整
                        blue += 1

                    if 0 < H < 180 and 0 < S < 255 and 0 < V < 46:
                        black += 1
                    elif 0 < H < 180 and 0 < S < 43 and 221 < V < 225:
                        white += 1
            color = "no"
            limit1 = limit2 = 0
            if yello * 2 >= card_img_count:
                color = "yello"
                limit1 = 11
                limit2 = 34  # 有的图片有色偏偏绿
            elif green * 4 >= card_img_count:
                if color == "yellow" and green < yello:
                    limit1 = 11
                    limit2 = 34  # 有的图片有色偏偏绿
                else:
                    color = "green"
                    limit1 = 35
                    limit2 = 99
            elif blue * 6 >= card_img_count:
                if color == "yellow" and blue < yello:
                    limit1 = 11
                    limit2 = 34  # 有的图片有色偏偏绿
                elif color == "green" and blue < green:
                    limit1 = 35
                    limit2 = 99
                else:
                    color = "blue"
                    limit1 = 100
                    limit2 = 124  # 有的图片有色偏偏紫
            elif black + white >= card_img_count * 0.7:  # TODO
                color = "bw"
            print('color is:', color)
            self.colors.append(color)
            # print('序号为:', t-1, blue, green, yello, black, white, card_img_count)
            # show_card = card_img.copy()
            # cv2.putText(show_card, color, (5,25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            # cv2.imshow("color"+str(t), show_card)
            # cv2.waitKey(0)
            if limit1 == 0:
                continue
            # 以上为确定车牌颜色
            # 以下为根据车牌颜色再定位，缩小边缘非车牌边界
            # cv2.imshow("color_hsv"+str(t), card_img_hsv)
            xl, xr, yh, yl = self.accurate_place(card_img_hsv, limit1, limit2, 100, color)
            if yl == yh and xl == xr:
                continue
            need_accurate = False
            if yl >= yh:
                yl = 0
                yh = row_num
                need_accurate = True
            if xl >= xr:
                xl = 0
                xr = col_num
                need_accurate = True
            # print(color)
            # print(yl,yh,xl,xr)
            if yl < 0: yl=0
            if yh < 0: yh=0
            if xl < 0: xl=0
            if xr < 0: xr=0
            # print('yes')
            self.card_imgs[card_index] = card_img[yl:yh, xl:xr] if color != "green" or yl < (yh - yl) // 4 else card_img[yl - (
                        yh - yl) // 4:yh, xl:xr]
            img_result_name = car_pic.split('.jpg', 1)[0] + '_result'
            self.card_num = car_pic.split('.jpg', 1)[0]
            # self.card_imgs_path[card_index]= img_result_name + '_' + str(card_index) + '.jpg'
            if color != "no":
                # cv2.imshow('result', self.card_imgs[card_index])
                cv2.imwrite(img_result_name+'.jpg', self.card_imgs[card_index])
            t+=1
        print('车牌定位结果保存到', img_result_name + '.jpg')
        print('time_use_locate:', time.time() - t0)
        # print('---------------------------------------------')
    # 以上为车牌定位
    # 以下为字符分割
    def split(self):
        self.part_cards = []
        t1 = time.time()
        for i, color in enumerate(self.colors):
            if color in ("blue", "yello", "green"):
                card_img = self.card_imgs[i]
                # card_img_path = self.card_imgs_path[i]
                print('---------------------------------------------')
                print('对上面图片进行车牌字符分割')
                gray_img = cv2.cvtColor(card_img, cv2.COLOR_BGR2GRAY)
                # 黄、绿车牌字符比背景暗、与蓝车牌刚好相反，所以黄、绿车牌需要反向
                if color == "green" or color == "yello":
                    gray_img = cv2.bitwise_not(gray_img)
                ret, gray_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                # cv2.imshow('split_thresh', gray_img)
                # 查找水平直方图波峰
                x_histogram = np.sum(gray_img, axis=1)
                x_min = np.min(x_histogram)
                x_average = np.sum(x_histogram) / x_histogram.shape[0]
                x_threshold = (x_min + x_average) / 2
                wave_peaks = find_waves(x_threshold, x_histogram)
                print("wave_peaks:", wave_peaks)
                if len(wave_peaks) == 0:
                    print("peak less 0:")
                    # continue
                # 认为水平方向，最大的波峰为车牌区域
                wave = max(wave_peaks, key=lambda x: x[1] - x[0])
                gray_img = gray_img[wave[0]:wave[1]]
                gray_img_old = gray_img.copy()
                # 查找垂直直方图波峰
                row_num, col_num = gray_img.shape[:2]
                # 去掉车牌上下边缘1个像素，避免白边影响阈值判断
                gray_img = gray_img[1:row_num - 1]
                y_histogram = np.sum(gray_img, axis=0)
                y_min = np.min(y_histogram)
                y_average = np.sum(y_histogram) / y_histogram.shape[0]
                y_threshold = (y_min + y_average) / 2.8  # U和0要求阈值偏小，否则U和0会被分成两半

                wave_peaks = find_waves(y_threshold, y_histogram)
                print("wave_peaks:", wave_peaks)
                # for wave in wave_peaks:
                #	cv2.line(card_img, pt1=(wave[0], 5), pt2=(wave[1], 5), color=(0, 0, 255), thickness=2)
                # 车牌字符数应大于6
                if len(wave_peaks) <= 6:
                    print("peak less 1:", len(wave_peaks))
                    # continue

                wave = max(wave_peaks, key=lambda x: x[1] - x[0])
                max_wave_dis = wave[1] - wave[0]
                # print("max_wave_dis:", max_wave_dis)
                # 判断是否是左侧车牌边缘
                if wave_peaks[0][1] - wave_peaks[0][0] < max_wave_dis / 3 and wave_peaks[0][0] == 0:
                    wave_peaks.pop(0)
                # print("wave_peaks:", wave_peaks)
                # 组合分离汉字(考虑偏旁|部首，需要合并)
                cur_dis = 0
                for i, wave in enumerate(wave_peaks):
                    if wave[1] - wave[0] + cur_dis > max_wave_dis * 0.75:
                        break
                    else:
                        cur_dis += wave[1] - wave[0]
                if i > 0:
                    wave = (wave_peaks[0][0], wave_peaks[i][1])
                    wave_peaks = wave_peaks[i + 1:]
                    wave_peaks.insert(0, wave)

                # 去除车牌上的分隔点
                point = wave_peaks[2]
                if point[1] - point[0] <= max_wave_dis / 2:
                    point_img = gray_img[:, point[0]:point[1]]
                    if np.mean(point_img) < 255 / 5:
                        wave_peaks.pop(2)

                if len(wave_peaks) <= 6:
                    print("peak less 2:", len(wave_peaks))
                    # continue
                # print("wave_peaks_after:", wave_peaks)
                self.part_cards = seperate_card(gray_img_old, wave_peaks)
                print('此车牌字符分割完成')
                # print('---------------------------------------------')
                print('time_use_split:', time.time()-t1)
                break
    # 以下为字符识别
    def recognize(self):
        self.predict_result = []
        t2 = time.time()
        print('---------------------------------------------')
        print('车牌字符识别结果为:')
        for i, part_card in enumerate(self.part_cards):
            # cv2.imshow("part_card_" + str(i), part_card)
            w = part_card.shape[1] // 3
            part_card = cv2.copyMakeBorder(part_card, 0, 0, w, w, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            part_card = cv2.resize(part_card, (SZ, SZ), interpolation=cv2.INTER_AREA)

            # cv2.imshow("part_card_" + str(i), part_card)
            part_card = deskew(part_card)
            # cv2.imshow("part_card_" + str(i), part_card)
            # part_card_name = self.card_num + '_part_' + str(i) + '.jpg'
            # cv2.imwrite(part_card_name, part_card)

            part_card = preprocess_hog([part_card])
            if i == 0:
                resp = self.modelchinese.predict(part_card)
                charactor = provinces[int(resp[0]) - PROVINCE_START]
            else:
                resp = self.model.predict(part_card)
                charactor = chr(resp[0])
            self.predict_result.append(charactor)
        print(self.predict_result)
        print('time_use_recognize:', time.time() - t2)


if __name__ == "__main__":
    # svm模型训练数据
    carpredictor_easy = CardPredictor(2)
    carpredictor_easy.train_svm()
    carpredictor_medium = CardPredictor(0)
    carpredictor_medium.train_svm()
    carpredictor_difficult = CardPredictor(1)
    carpredictor_difficult.train_svm()
    # locate test
    # 1-2: 6 and G(wrong) | 1-2: 1 and 2(wrong) | 1-2: 0(wrong) and D
    # 3-1 and 3-3: 1 and 7(wrong) | 3-2: 6(wrong) and 5 | 3-3: 3 and J(wrong)
    # carpredictor.locate('./images/easy/2-3.jpg')
    carpredictor_easy.locate('./images/easy/1-1.jpg')
    carpredictor_easy.split()
    carpredictor_easy.recognize()
    carpredictor_easy.locate('./images/easy/1-2.jpg')
    carpredictor_easy.split()
    carpredictor_easy.recognize()
    carpredictor_easy.locate('./images/easy/1-3.jpg')
    carpredictor_easy.split()
    carpredictor_easy.recognize()
    carpredictor_medium.locate('./images/medium/2-1.jpg')
    carpredictor_medium.split()
    carpredictor_medium.recognize()
    carpredictor_medium.locate('./images/medium/2-2.jpg')
    carpredictor_medium.split()
    carpredictor_medium.recognize()
    carpredictor_medium.locate('./images/medium/2-3.jpg')
    carpredictor_medium.split()
    carpredictor_medium.recognize()
    carpredictor_difficult.locate('./images/difficult/3-1.jpg')
    carpredictor_difficult.split()
    carpredictor_difficult.recognize()
    carpredictor_difficult.locate('./images/difficult/3-2.jpg')
    carpredictor_difficult.split()
    carpredictor_difficult.recognize()
    carpredictor_difficult.locate('./images/difficult/3-3.jpg')
    carpredictor_difficult.split()
    carpredictor_difficult.recognize()

    while True:
        k = cv2.waitKey(1)
        if k == ord('q'):
            cv2.destroyAllWindows()
            break
