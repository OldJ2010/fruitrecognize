import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import sys

# sys.path.append("C:/projects/workspace/fruit")
import mylib

# 把配置参数统一放到这里来配置
config_canny_low_thresh = 100
config_canny_high_thresh = 160


# 计算一个区域是否为疑似水果的区域
def is_area_valid_simple(topleft_x, topleft_y, area_width, area_height, area_sum):
    if (area_height < 10) and (area_width < 10):
        return False
    if (area_height > 600) and (area_width >400):
        return False
    if (area_sum < 300):
        return False
    if (area_height > area_width) and (area_height / area_width > 2):
        return False
    if (area_height < area_width) and (area_width / area_height > 2):
        return False
    if (area_sum * 3) < (area_width * area_height * 2):
        return False
    return True


# 开始，读入图片
img_original = cv2.imread('fruits.png')
img_factor = 1
img_dest = img_original
img_dest = cv2.pyrDown(img_dest)  # 取到1/2原图像
img_factor *= 2
# img_dest = cv2.pyrDown(img_dest)  # 取到1/4原图像
# img_factor *= 2

imghsv = cv2.cvtColor(img_dest, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(imghsv)

edges_dest = cv2.Canny(img_dest, config_canny_low_thresh, config_canny_high_thresh)
edges_s = cv2.Canny(s, config_canny_low_thresh, config_canny_high_thresh)
edges_detail = cv2.bitwise_or(edges_dest, edges_s)
height, width = edges_detail.shape

kernel_closing = np.ones((3, 3), np.uint8)
edges_closing = cv2.morphologyEx(edges_detail, cv2.MORPH_CLOSE, kernel_closing)
edges_detail = edges_closing

imgBlock = np.zeros((height, width, 3), np.uint8)

edges_inv = cv2.bitwise_not(edges_detail)
output = cv2.connectedComponentsWithStats(edges_inv, connectivity=4)
nLabels = output[0]
labelImage = output[1]
stats = output[2]

print("连通域个数=%d" % (nLabels))
# print("LabelImage=%s" % (labelImage,))
# print("Stats=%s" % (stats,))

# 根据stats结果，只有满足面积、形状、颜色等条件的才是水果
fruit_counter = 0
colorKernel = np.zeros((nLabels), np.uint8)
colors = []
for i in range(1, nLabels + 1):
    colors.append(np.array([random.randint(20, 255), random.randint(20, 255), random.randint(20, 255)]))

img_fruit = []
img_fruit_original = []
for index in range(nLabels):
    img_fruit.append([])
    img_fruit_original.append([])
    areastat = stats[index]
    #print("stats[%d]=%s" % (index, areastat,))
    topleft_x, topleft_y, area_width, area_height, area_sum = areastat
    # 判断物体大小，可以参考透视，远小进大的原则，需要把透视增加进去
    if not is_area_valid_simple(topleft_x, topleft_y, area_width, area_height, area_sum):
        continue

    # 取出颜色值进行特征挖掘
    # 把找出的水果放在这里，包括左上角坐标，起点位置，高和宽，以外接矩形的方式表示，从stats中获取
    #
    imgFruitMask = np.zeros((area_height, area_width), np.uint8)
    for masky in range(area_height):
        for maskx in range(area_width):
            if labelImage[topleft_y + masky, topleft_x + maskx] == index:
                imgFruitMask[masky, maskx] = 255

    ball = img_dest[topleft_y:topleft_y + area_height, topleft_x:topleft_x + area_width]
    imgFruitIndex = np.zeros((area_height, area_width), np.uint8)
    imgFruitIndex = ball
    img_fruit[index] = cv2.bitwise_and(imgFruitIndex, imgFruitIndex, mask=imgFruitMask)

    tly = topleft_y * img_factor
    tlx = topleft_x * img_factor
    area_height_original = area_height * img_factor
    area_width_original = area_width * img_factor
    ballOriginal = img_original[tly:tly + area_height_original, tlx:tlx + area_width_original]
    imgFruitIndexOriginal = np.zeros((area_height_original, area_width_original), np.uint8)
    imgFruitIndexOriginal = ballOriginal
    imgFruitMaskOriginal = cv2.resize(imgFruitMask, None, fx=img_factor, fy=img_factor, interpolation=cv2.INTER_AREA)
    img_fruit_original[index] = cv2.bitwise_and(imgFruitIndexOriginal, imgFruitIndexOriginal, mask=imgFruitMaskOriginal)

    print("fruit label:%d" % index)
    colorKernel[index] = 255
    fruit_counter += 1

print("共发现可疑水果个数：%d" % fruit_counter)
for y in range(height):
    for x in range(width):
        label = labelImage[y, x]
        if (label > 0) and (colorKernel[label] > 0):
            if stats[label][4] > 100:
                imgBlock[y, x] = colors[label]

fruit_random = random.randint(1, fruit_counter)
print("随机号码为：%d" % fruit_random)
hist_counter = 0
needExit = False
for fruitIndex in range(nLabels):
    if colorKernel[fruitIndex] == 255:
        hist_counter += 1
        if hist_counter == fruit_random:
            img_4hist_bgr = img_fruit_original[fruitIndex]
            b, g, r = cv2.split(img_4hist_bgr)
            # 统计的时候要把黑色的去除
            histImgB = mylib.calcAndDrawHist(b, [255, 0, 0])
            histImgG = mylib.calcAndDrawHist(g, [0, 255, 0])
            histImgR = mylib.calcAndDrawHist(r, [0, 0, 255])
            # cv2.imshow("b", histImgB)
            # cv2.imshow("g", histImgG)
            # cv2.imshow("r", histImgR)
            cv2.imshow('fruitIMG', img_4hist_bgr)
            needExit = True

    if needExit:
        break

# imghsv = cv2.cvtColor(imgFruit[161], cv2.COLOR_BGR2HSV)
# imghsv = imgFruit[246]
# h, s, v = cv2.split(imghsv)
# histImgB = mylib.calcAndDrawHist(h, [255, 0, 0])
# histImgG = mylib.calcAndDrawHist(s, [0, 255, 0])
# histImgR = mylib.calcAndDrawHist(v, [0, 0, 255])
#
# # imgCombine = imgBlock + dst
# cv2.imshow('imgFruit232', imgFruit[161])
# cv2.imshow('imgFruit246', imgFruit[246])
# cv2.imshow("H", histImgB)
# cv2.imshow("S", histImgG)
# cv2.imshow("V", histImgR)

# cv2.imshow('edgesdownup', cv2.pyrUp(cv2.pyrDown(edges)))
#
# cv2.imshow('edges', edges)
# cv2.imshow('img', img_dest)
cv2.imshow('imgBlock', imgBlock)
cv2.imshow('edges_closing', edges_closing)
imgNew = cv2.add(img_dest, imgBlock)
cv2.imshow('imgNew', imgNew)

# cv2.imwrite('/Users/lizheng/Test/fruit/img/edges5.png',edges)
# cv2.imshow('edges', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
# np.savetxt("/Users/lizheng/Test/fruit/pythonProject/numpy/fruittest_labelblock.txt", labelImage, fmt='%s')
# cv2.imwrite('/Users/lizheng/Test/fruit/img/block.png',imgBlock)
