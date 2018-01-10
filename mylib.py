from numpy import *
import math
import numpy as np
import cv2


def calc_rect(p1_x, p1_y, p2_x, p2_y):
    yp1 = p1_y
    xp1 = p1_x
    yp2 = p2_y
    xp2 = p2_x
    lenAB = math.sqrt(math.pow(xp1 - xp2, 2) + math.pow(yp1 - yp2, 2))

    print("Length=%f" % lenAB)

    totalLength = 200
    delta_xp = (math.fabs(xp2 - xp1) * (totalLength - lenAB) / 2) / lenAB
    delta_yp = (math.fabs(yp1 - yp2) * (totalLength - lenAB) / 2) / lenAB

    if (xp1 > xp2):
        kx = 1
    else:
        kx = -1

    if (yp1 > yp2):
        ky = 1
    else:
        ky = -1

    xp1_0 = xp1 + kx * delta_xp
    yp1_0 = yp1 + ky * delta_yp
    xp2_0 = xp2 - kx * delta_xp
    yp2_0 = yp2 - ky * delta_yp

    print("deltaXY=[%f,%f]" % (delta_xp, delta_yp))
    print("p1=[%f,%f], p2=[%f,%f]" % (xp1_0, yp1_0, xp2_0, yp2_0))

    intp1x = np.int0(xp1_0)
    intp1y = np.int0(yp1_0)
    intp2x = np.int0(xp2_0)
    intp2y = np.int0(yp2_0)
    print("point p1=[%d,%d], p2=[%d,%d]" % (intp1x, intp1y, intp2x, intp2y))

    return intp1x, intp1y, intp2x, intp2y


# 获取一个点阵集描述的曲线的两个端点，如果是封闭的，则不返回
def getTerminalPoint(x, y, edges):
    height, width = np.shape(edges)
    counter = 0
    if (y > 0) and (x > 0) and (edges[y - 1, x - 1] > 0):
        counter = counter + 1
    if (y > 0) and (edges[y - 1, x] > 0):
        counter = counter + 1
    if (y > 0) and (x < width) and (edges[y - 1, x + 1] > 0):
        counter = counter + 1
    if (x > 0) and (edges[y, x - 1] > 0):
        counter = counter + 1
    if (x < width) and (edges[y, x + 1] > 0):
        counter = counter + 1
    if (x > 0) and (y < height) and (edges[y + 1, x - 1] > 0):
        counter = counter + 1
    if (y < height) and (edges[y + 1, x] > 0):
        counter = counter + 1
    if (y < height) and (x < width) and (edges[y + 1, x + 1] > 0):
        counter = counter + 1
    return counter


# 找圆的方法, 在相连的点内，不跨不相连的点
# listPoints 连接的线的集合
# minRadius 最小的半径
# maxRadius 最大的半径 系统限制为100
# width 线宽
# minDistance 圆心之间的最小距离
# imgWidth 图像的宽
# imgHeight 图像的高
# minPoints 最少包含的点
def findCircle(listPoints, minRadius, maxRadius, width, minDistance, imgWidth, imgHeight, minPoints):
    imgLabel = np.zeros((imgHeight, imgWidth), np.uint8)
    for y in range(imgHeight):
        print("total=%d, process y=%d" % (imgHeight, y))
        for x in range(imgWidth):
            for radius in range(minRadius, maxRadius):
                imgTemp = np.zeros((imgHeight, imgWidth), np.uint8)
                cv2.circle(imgTemp, (x, y), radius, 255, width)  # 画圆
                for points in listPoints:
                    if len(points) < minPoints:
                        continue
                    for x, y in points:
                        if imgTemp[y, x] > 0:
                            imgLabel[y, x] += 1

    return imgLabel


# 计算点的连通性
def connect(edges, x, y):
    height, width = edges.shape
    height = height - 1
    width = width - 1
    counter = 0
    if (y > 0) and (x > 0) and (edges[y - 1, x - 1] > 0):
        counter = counter + 1
    if (y > 0) and (edges[y - 1, x] > 0):
        counter = counter + 1
    if (y > 0) and (x < width) and (edges[y - 1, x + 1] > 0):
        counter = counter + 1
    if (x > 0) and (edges[y, x - 1] > 0):
        counter = counter + 1
    if (x < width) and (edges[y, x + 1] > 0):
        counter = counter + 1
    if (x > 0) and (y < height) and (edges[y + 1, x - 1] > 0):
        counter = counter + 1
    if (y < height) and (edges[y + 1, x] > 0):
        counter = counter + 1
    if (y < height) and (x < width) and (edges[y + 1, x + 1] > 0):
        counter = counter + 1
    return counter


# 判断一个曲线是否是凸的
# 在一个曲线的两端，在一个规定的矩形范围内，进行概率式的延展拟合和其他的曲线，找出一个凸的图形
# calc_rect(549., 58.00000763, 521., 132.)


# histogram统计
def calcAndDrawHist(image, color):
    hist = cv2.calcHist([image], [0], None, [256], [1.0, 255.0])
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(hist)
    histImg = np.zeros([256, 256, 3], np.uint8)
    hpt = int(0.9 * 256)

    for h in range(256):
        intensity = int(hist[h] * hpt / maxVal)
        cv2.line(histImg, (h, 256), (h, 256 - intensity), color)

    return histImg
