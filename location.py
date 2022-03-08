import cv2
import numpy as np
from numpy.linalg import norm
import sys
import os
import json
# # for difficult
MAX_WIDTH_DIF = 500
# # for medium
MAX_WIDTH_MED = 450

SZ = 20          #训练图片长宽
# MAX_WIDTH = 500 #原始图片最大宽度
Min_Area = 1000  #车牌区域允许最小面积


def imreadex(filename):
    return cv2.imdecode(np.fromfile(filename, dtype=np.uint8), cv2.IMREAD_COLOR)


def point_limit(point):
    if point[0] < 0:
        point[0] = 0
    if point[1] < 0:
        point[1] = 0

class CardPredictor:
    def __init__(self, type):
        self.type = type
        if type:
            self.MAX_WIDTH = MAX_WIDTH_DIF
        else:
            self.MAX_WIDTH = MAX_WIDTH_MED
        # 车牌识别的部分参数保存在js中，便于根据图片分辨率做调整
        f = open('config.json')
        j = json.load(f)
        for c in j["config"]:
            if c["open"]:
                self.cfg = c.copy()
                break
        else:
            raise RuntimeError('没有设置有效配置参数')

    def accurate_place(self, card_img_hsv, limit1, limit2, color):
        row_num, col_num = card_img_hsv.shape[:2]
        print("row, col num:", row_num, col_num)
        xl = col_num
        xr = 0
        yh = 0
        yl = row_num
        # col_num_limit = self.cfg["col_num_limit"]
        row_num_limit = self.cfg["row_num_limit"]
        col_num_limit = col_num * 0.8 if color != "green" else col_num * 0.3  # 绿色有渐变
        for i in range(row_num):
            count = 0
            for j in range(col_num):
                H = card_img_hsv.item(i, j, 0)
                S = card_img_hsv.item(i, j, 1)
                V = card_img_hsv.item(i, j, 2)
                if limit1 < H <= limit2 and 34 < S and 46 < V:
                    count += 1
            if count > col_num_limit:
                if yl > i:
                    yl = i
                if yh < i:
                    yh = i
        for j in range(col_num):
            count = 0
            for i in range(row_num):
                H = card_img_hsv.item(i, j, 0)
                S = card_img_hsv.item(i, j, 1)
                V = card_img_hsv.item(i, j, 2)
                if limit1 < H <= limit2 and 34 < S and 46 < V:
                    count += 1
            if count > row_num - row_num_limit:
                if xl > j:
                    xl = j
                if xr < j:
                    xr = j
        return xl, xr, yh+2, yl-2

    def locate(self, car_pic, resize_rate=1):
        if type(car_pic) == type(""):
            img = imreadex(car_pic)
        else:
            img = car_pic
        pic_hight, pic_width = img.shape[:2]
        if pic_width > self.MAX_WIDTH:
            pic_rate = self.MAX_WIDTH / pic_width
            img = cv2.resize(img, (self.MAX_WIDTH, int(pic_hight * pic_rate)), interpolation=cv2.INTER_LANCZOS4)

        if resize_rate != 1:
            img = cv2.resize(img, (int(pic_width * resize_rate), int(pic_hight * resize_rate)),
                             interpolation=cv2.INTER_LANCZOS4)
            pic_hight, pic_width = img.shape[:2]
        cv2.imshow('original', img)
        # print("h,w:", pic_hight, pic_width)
        # pic_hight, pic_width = img.shape[:2]
        print("h,w:", img.shape)
        blur = self.cfg["blur"]
        oldimg = img
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 高斯去噪
        if blur > 0:
            img = cv2.GaussianBlur(img, (blur, blur), 0)  # 图片分辨率调整
        # 中值滤波
        median = cv2.medianBlur(img, 5)
        # cv2.imshow('gray', img)
        # 去掉图像中不会是车牌的区域
        kernel = np.ones((20, 20), np.uint8)
        img_opening = cv2.morphologyEx(median, cv2.MORPH_OPEN, kernel)
        img_opening = cv2.addWeighted(median, 1, img_opening, -1, 0)
        # cv2.imshow('addweight', img_opening)
        # for difficult
        if type:
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
        cv2.imshow('edge1', img_edge1)
        img_edge2 = cv2.morphologyEx(img_edge1, cv2.MORPH_OPEN, kernel)
        cv2.imshow('edge2', img_edge2)

        # 查找图像边缘整体形成的矩形区域，可能有很多，车牌就在其中一个矩形区域中
        try:
            contours, hierarchy = cv2.findContours(img_edge2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        except ValueError:
            image, contours, hierarchy = cv2.findContours(img_edge2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = [cnt for cnt in contours if cv2.contourArea(cnt) > Min_Area]
        print('len(contours)', len(contours))
        # print(contours)
        # cv2.drawContours(oldimg, contours, 0, (0, 0, 255), 2)
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
            if wh_ratio > 1.5 and wh_ratio < 7.5:
                car_contours.append(rect)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                old_img = cv2.drawContours(old_img, [box], 0, (0, 0, 255), 2)
        cv2.imshow("edge4", old_img)
            # cv2.waitKey(0)

        print(len(car_contours))

        print("精确定位")
        card_imgs = []
        # 矩形区域可能是倾斜的矩形，需要矫正，以便使用颜色定位
        i = 0
        # print("car_contours:", car_contours)
        for rect in car_contours:
            if rect[2] > -1 and rect[2] < 1:  # 创造角度，使得左、高、右、低拿到正确的值
                angle = 1
            else:
                angle = rect[2]
            rect = (rect[0], (rect[1][0] + 5, rect[1][1] + 5), angle)  # 扩大范围，避免车牌边缘被排除
            # print("rect", rect)

            box = cv2.boxPoints(rect)
            # print("box", box)
            heigth_point = right_point = [0, 0]
            left_point = low_point = [pic_width, pic_hight]
            print(left_point, low_point, right_point, heigth_point)
            for point in box:
                if left_point[0] > point[0]:
                    left_point = point
                if low_point[1] > point[1]:
                    low_point = point
                if heigth_point[1] < point[1]:
                    heigth_point = point
                if right_point[0] < point[0]:
                    right_point = point
            print(left_point, low_point, right_point, heigth_point)
            # cv2.imshow('oldimg_', oldimg)
            if left_point[1] <= right_point[1]:  # 正角度
                new_right_point = [right_point[0], heigth_point[1]]
                pts2 = np.float32([left_point, heigth_point, new_right_point])  # 字符只是高度需要改变
                pts1 = np.float32([left_point, heigth_point, right_point])
                # print("pts1, pts2:", pts1, '\n', pts2)
                M = cv2.getAffineTransform(pts1, pts2)
                dst = cv2.warpAffine(oldimg, M, (pic_width, pic_hight))
                # cv2.imshow('dst', dst)
                point_limit(new_right_point)
                point_limit(heigth_point)
                point_limit(left_point)
                card_img = dst[int(left_point[1]):int(heigth_point[1]), int(left_point[0]):int(new_right_point[0])]
                card_imgs.append(card_img)
            # cv2.imshow("card", card_img)
            # cv2.waitKey(0)
            elif left_point[1] > right_point[1]:  # 负角度

                new_left_point = [left_point[0], heigth_point[1]]
                pts2 = np.float32([new_left_point, heigth_point, right_point])  # 字符只是高度需要改变
                pts1 = np.float32([left_point, heigth_point, right_point])
                M = cv2.getAffineTransform(pts1, pts2)
                dst = cv2.warpAffine(oldimg, M, (pic_width, pic_hight))
                # cv2.imshow('dst', dst)
                point_limit(right_point)
                point_limit(heigth_point)
                point_limit(new_left_point)
                card_img = dst[int(right_point[1]):int(heigth_point[1]), int(new_left_point[0]):int(right_point[0])]
                card_imgs.append(card_img)
            # cv2.imshow("card"+str(i), card_img)
            # cv2.waitKey(0)
            i+=1
        # 开始使用颜色定位，排除不是车牌的矩形，目前只识别蓝、绿、黄车牌
        colors = []
        j = 0
        for card_index, card_img in enumerate(card_imgs):
            green = yello = blue = black = white = 0
            card_img_hsv = cv2.cvtColor(card_img, cv2.COLOR_BGR2HSV)
            cv2.imshow('card_img'+str(j), card_img)
            j+=1
            # 有转换失败的可能，原因来自于上面矫正矩形出错
            if card_img_hsv is None:
                continue
            row_num, col_num = card_img_hsv.shape[:2]
            card_img_count = row_num * col_num

            for i in range(row_num):
                for j in range(col_num):
                    H = card_img_hsv.item(i, j, 0)
                    S = card_img_hsv.item(i, j, 1)
                    V = card_img_hsv.item(i, j, 2)
                    if 11 < H <= 34 and S > 34:  # 图片分辨率调整
                        yello += 1
                    elif 35 < H <= 99 and S > 34:  # 图片分辨率调整
                        green += 1
                    elif 99 < H <= 124 and S > 34:  # 图片分辨率调整
                        blue += 1

                    if 0 < H < 180 and 0 < S < 255 and 0 < V < 46:
                        black += 1
                    elif 0 < H < 180 and 0 < S < 43 and 221 < V < 225:
                        white += 1
            color = "no"

            limit1 = limit2 = 0
            if yello * 3 >= card_img_count:
                color = "yello"
                limit1 = 11
                limit2 = 34  # 有的图片有色偏偏绿
            elif green * 4 >= card_img_count:
                color = "green"
                limit1 = 35
                limit2 = 99
            elif blue * 3 >= card_img_count:
                color = "blue"
                limit1 = 100
                limit2 = 124  # 有的图片有色偏偏紫
            elif black + white >= card_img_count * 0.7:  # TODO
                color = "bw"
            print(color)
            colors.append(color)
            print(blue, green, yello, black, white, card_img_count)
            # cv2.imshow("color", card_img)
            # cv2.waitKey(0)
            if limit1 == 0:
                continue
            # 以上为确定车牌颜色
            # 以下为根据车牌颜色再定位，缩小边缘非车牌边界
            cv2.imshow("color_hsv", card_img_hsv)
            xl, xr, yh, yl = self.accurate_place(card_img_hsv, limit1, limit2, color)
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
            print(color)
            print(yl,yh,xl,xr)
            if yl < 0: yl=0
            if yh < 0: yh=0
            if xl < 0: xl=0
            if xr < 0: xr=0
            card_imgs[card_index] = card_img[yl:yh, xl:xr] if color != "green" or yl < (yh - yl) // 4 else card_img[yl - (
                        yh - yl) // 4:yh, xl:xr]
            if color != "no":
                # print(card_imgs[card_index])
                cv2.imshow("final_img", card_imgs[card_index])
                cv2.imwrite('final_img.jpg', card_imgs[card_index])
            if need_accurate:  # 可能x或y方向未缩小，需要再试一次
                card_img = card_imgs[card_index]
                card_img_hsv = cv2.cvtColor(card_img, cv2.COLOR_BGR2HSV)
                xl, xr, yh, yl = self.accurate_place(card_img_hsv, limit1, limit2, color)
                if yl == yh and xl == xr:
                    continue
                if yl >= yh:
                    yl = 0
                    yh = row_num
                if xl >= xr:
                    xl = 0
                    xr = col_num
            card_imgs[card_index] = card_img[yl:yh, xl:xr] if color != "green" or yl < (yh - yl) // 4 else card_img[yl - (
                        yh - yl) // 4:yh, xl:xr]
            # cv2.imshow("final", card_img)
    # 以上为车牌定位

if __name__ == "__main__":
    # carpredictor_medium = CardPredictor(0)
    carpredictor_difficult = CardPredictor(1)
    # carpredictor.locate('./images/easy/2-3.jpg')
    # carpredictor_medium.locate('./images/medium/2-3.jpg')
    carpredictor_difficult.locate('./images/difficult/3-1.jpg')
    # carpredictor.locate('./test/car4.jpg')
    while True:
        k = cv2.waitKey(1)
        if k == ord('q'):
            cv2.destroyAllWindows()
            break
