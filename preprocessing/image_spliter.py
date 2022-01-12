import cv2
import os
import sys
import numpy as np
from tqdm import tqdm
from PIL import Image

#### def ####
import seaborn as sns
import matplotlib.pyplot as plt

import matplotlib.image as mpimg
from matplotlib import font_manager, rc
rc('font',family="AppleGothic")
plt.rcParams["font.family"]="AppleGothic" #plt 한글꺠짐
plt.rcParams["font.family"]="Arial" #외국어꺠짐
plt.rcParams['axes.unicode_minus'] = False # 마이너스 부호 출력 설정
plt.rc('figure', figsize=(10,8))

sns.set(font="AppleGothic", 
        rc={"axes.unicode_minus":False},
        style='darkgrid') #sns 한글깨짐

from tqdm import tqdm


def show(img):
    #사이즈
    plt.figure(figsize = (100,100))
    #xticks/yticks - 눈금표
    plt.xticks([])
    plt.yticks([])
    #코랩에서 안돌아감 주의
    plt.imshow(img, cmap= 'gray')
    plt.show()

def histogram_equalization(img):
    hist, bins = np.histogram(img.flatten(), 256,[0,256])

    cdf = hist.cumsum()

    # cdf의 값이 0인 경우는 mask처리를 하여 계산에서 제외
    # mask처리가 되면 Numpy 계산에서 제외가 됨
    # 아래는 cdf array에서 값이 0인 부분을 mask처리함
    cdf_m = np.ma.masked_equal(cdf,0)

    #History Equalization 공식
    cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())

    # Mask처리를 했던 부분을 다시 0으로 변환
    cdf = np.ma.filled(cdf_m,0).astype('uint8')

    img2 = cdf[img]

    return img2

def get_crop_images(image_origin, contours):
    margin = 10  # 원하는 margin
    image_copy = image_origin.copy()
    origin_height, origin_width = image_copy.shape[:2]  # get image size

    crop_images = []  # 자른 이미지를 하나씩 추가해서 저장할 리스트

    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)  # 좌상단 꼭지점 좌표 , width, height

        # Rect 의 size 가 기준 이상인 것만 담는다
        if width > 1000 and height > 1000:
            crop_row_1 = (y - margin) if (y - margin) > 0 else y
            crop_row_2 = (y + height + margin) if (y + height + margin) < origin_height else y + height
            crop_col_1 = (x - margin) if (x - margin) > 0 else x
            crop_col_2 = (x + width + margin) if (x + width + margin) < origin_width else x + width

            # 행렬은 row col 순서!!! 햇갈리지 말자!
            crop = image_copy[crop_row_1: crop_row_2, crop_col_1: crop_col_2]  # 이미지를 잘라낸다.
            crop_images.append(crop)  # 잘라낸 이미지들을 하나씩 리스트에 담는다.
            

    return crop_images

def convertImage(image_name, save_name):
	img = Image.open(image_name)
	img = img.convert("RGBA")

	datas = img.getdata()

	newData = []

	for items in datas:
		if items[0] == 255 and items[1] == 255 and items[2] == 255:
			newData.append((255, 255, 255, 0))
		else:
			newData.append(items)

	img.putdata(newData)
	img.save("only_edge_{}.jpg".format(save_name), "PNG")
	print("Successful")




##### image_extraction #####
def get_ready(img_org_path, save_name):
    img_org = cv2.imread(img_org_path, cv2.IMREAD_COLOR)
    img = img_org.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)

    #블러 02
    #블러의 커널 사이즈가 홀수만 가능하므로 이미지 평균 값을 기준으로 홀수값 만들기
    blur_k = int((img.mean()*0.5)//2)*2+1 
    img = cv2.medianBlur(img, blur_k)

    #threshold 적용을 위해 Lab에서 Grayscale로 이미지 변환 03
    img = cv2.cvtColor(img, cv2.COLOR_Lab2BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #이미지 평균값을 기준으로 이진화 04
    ret, img = cv2.threshold(img, 30, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


    ######## extracting screen #######
    for num, i in enumerate(get_crop_images(img_org, contours)[2:3]):
        cv2.imwrite('screen'+'_{}'.format(save_name)+'.jpg', i)
        print('screen extraction complete')


    ### screen masking
    screen_mask = img_org.copy()
    cv2.drawContours(screen_mask, contours.copy(), -1, (255,255,255), -1)
    cv2.drawContours(screen_mask, contours.copy(), 2, (255,255,255), -1)

    ### edge
    for num, i in enumerate(get_crop_images(screen_mask, contours)[1:2]):
        # show(i)
        white_img = i.copy()
        cv2.imwrite('edge_only'+'_{}'.format(save_name)+'.jpg', i)
        print('edge extraction complete')
        image_name = 'edge_only'+'_{}'.format(save_name)+'.jpg'

    ### edge png
    convertImage(image_name, save_name)
    print('background removed')