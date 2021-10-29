# -*- coding: utf-8 -*-
import numpy as np
import json
import cv2
import random
import os
import sys
from PIL import Image



def cvminAreaRect2longsideformat(x_c, y_c, width, height, theta):
    '''
    trans minAreaRect(x_c, y_c, width, height, θ) to longside format(x_c, y_c, longside, shortside, θ)
    两者区别为:
            当opencv表示法中width为最长边时（包括正方形的情况），则两种表示方法一致
            当opencv表示法中width不为最长边 ，则最长边表示法的角度要在opencv的Θ基础上-90度
    @param x_c: center_x
    @param y_c: center_y
    @param width: x轴逆时针旋转碰到的第一条边
    @param height: 与width不同的边
    @param theta: x轴逆时针旋转与width的夹角，由于原点位于图像的左上角，逆时针旋转角度为负 [-90, 0)
    @return:
            x_c: center_x
            y_c: center_y
            longside: 最长边
            shortside: 最短边
            theta_longside: 最长边和x轴逆时针旋转的夹角，逆时针方向角度为负 [-180, 0)
    '''
    '''
    意外情况:(此时要将它们恢复符合规则的opencv形式：wh交换，Θ置为-90)
    竖直box：box_width < box_height  θ=0
    水平box：box_width > box_height  θ=0
    '''

    if theta == 0:
        theta = -90
        buffer_width = width
        width = height
        height = buffer_width

    if theta > 0:
        if theta != 90:  # Θ=90说明wh中有为0的元素，即gt信息不完整，无需提示异常，直接删除
            print('θ计算出现异常，当前数据为：%.16f, %.16f, %.16f, %.16f, %.1f;超出opencv表示法的范围：[-90,0)' % (
            x_c, y_c, width, height, theta))
        return False

    if theta < -90:
        print(
            'θ计算出现异常，当前数据为：%.16f, %.16f, %.16f, %.16f, %.1f;超出opencv表示法的范围：[-90,0)' % (x_c, y_c, width, height, theta))
        return False

    if width != max(width, height):  # 若width不是最长边
        longside = height
        shortside = width
        theta_longside = theta - 90
    else:  # 若width是最长边(包括正方形的情况)
        longside = width
        shortside = height
        theta_longside = theta

    if longside < shortside:
        print('旋转框转换表示形式后出现问题：最长边小于短边;[%.16f, %.16f, %.16f, %.16f, %.1f]' % (
        x_c, y_c, longside, shortside, theta_longside))
        return False
    if (theta_longside < -180 or theta_longside >= 0):
        print('旋转框转换表示形式时出现问题:θ超出长边表示法的范围：[-180,0);[%.16f, %.16f, %.16f, %.16f, %.1f]' % (
        x_c, y_c, longside, shortside, theta_longside))
        return False

    return x_c, y_c, longside, shortside, theta_longside


def get_minrec(file_dir, img_dir, yolo_dir, labelname, class_id):
    print("labelname", labelname)

    filelist = os.listdir(file_dir)
    imglist = os.listdir(img_dir)
    print("filelist", filelist)
    print("imglist", imglist)
    for file_json in filelist:
        img_name = file_json.split(".json")[0].split('-')[0]

        img_fullname = os.path.join(img_dir, img_name + ".jpg")
        img_fulljson = os.path.join(file_dir, file_json)
        # print("img_name",img_fullname)
        with open(img_fulljson, 'r', encoding='UTF-8') as f:
            data = json.load(f)

            img = cv2.imread(img_fullname)
            # print("img.shape",img.shape)
            codinate = []
            codinate_name=[]
            for label in data["shapes"]:
                #print("label",label)
                if(label["label"]=='gkxfw1'):
                    label["label"]='gkxfw'
                if (label["label"] in labelname):
                    codinate_name.append(class_id[label["label"]])
                    codinate.append(np.int32(label["points"]))
            # print("codinate",codinate)
            # print("codinate_name", codinate_name)

            if not os.path.exists(yolo_dir):  # 判断是否存在文件夹如果不存在则创建为文件夹
                os.makedirs(yolo_dir)

            with open(os.path.join(yolo_dir, img_name + '.txt'), 'w') as f_out:
                num_gt = 0
                for _, (i,name_id) in enumerate(zip(codinate,codinate_name)):
                    # print("i",i)
                    # print('name_id',name_id)

                    num_gt = num_gt + 1
                    # print("i",i)
                    if(name_id=='0'or name_id=='1' or name_id=='2' ):
                        rect = cv2.minAreaRect(i)
                        points = cv2.boxPoints(rect)
                        # print("points",np.int32(points))
                        # max_shape = max(img.shape[1], img.shape[0])
                        points[:, 0] = points[:, 0]
                        points[:, 1] = points[:, 1]

                        rect = cv2.minAreaRect(points)
                        # print("rect",rect)
                        c_x = rect[0][0]
                        c_y = rect[0][1]
                        w = rect[1][0]
                        h = rect[1][1]
                        theta = rect[-1]  # Range for angle is [-90，0)
                        trans_data = cvminAreaRect2longsideformat(c_x, c_y, w, h, theta)
                        if not trans_data:
                            if theta != 90:  # Θ=90说明wh中有为0的元素，即gt信息不完整，无需提示异常，直接删除
                                print('opencv表示法转长边表示法出现异常,已将第%d个box排除,问题出现在该图片中:%s' % (_, img_fullname))
                            num_gt = num_gt - 1
                            continue

                    else:
                        max_shape = max(img.shape[1], img.shape[0])
                        # print("rect",rect)
                        c_x = ((i[0][0] + i[1][0]) / 2)
                        c_y = ((i[0][1] + i[1][1]) / 2)
                        w = abs((i[1][0] - i[0][0]))
                        h = abs((i[1][1] - i[0][1]))
                        points=[[c_x-w/2,c_y-h/2],[c_x+w/2,c_y-h/2],[c_x+w/2,c_y+h/2],[c_x-w/2,c_y+h/2]]


                    # if not trans_data:
                    #     if theta != 90:  # Θ=90说明wh中有为0的元素，即gt信息不完整，无需提示异常，直接删除
                    #         print('opencv表示法转长边表示法出现异常,已将第%d个box排除,问题出现在该图片中:%s' % (_, img_fullname))
                    #     num_gt = num_gt - 1
                    #     continue
                    # else:
                    #     # range:[-180，0)
                    #     c_x, c_y, longside, shortside, theta_longside = trans_data
                    # # print("theta_longside",theta_longside)
                    # bbox = np.array((c_x/img.shape[1], c_y/img.shape[0], longside/img.shape[1], shortside/img.shape[0]))
                    points=np.array(points)
                    points[:, 0] = points[:, 0] / img.shape[1]#width
                    points[:, 1] = points[:, 1] / img.shape[0]#height
                    bbox = points


                    if (bbox.any() <0) or (bbox.any() >1):  # 0<xy<1, 0<side<=1
                        print('bbox中有>= 1的元素,bbox中有<= 0的元素,已将第%d个box排除,问题出现在该图片中:%s' % (i, img_fullname))
                        print('出问题的longside形式数据:',bbox)
                        num_gt = num_gt - 1
                        continue
                    # theta_label = int(theta_longside + 180.5)  # range int[0,180] 四舍五入
                    # if theta_label == 180:  # range int[0,179]
                    #     theta_label = 179
                    # if theta_label < 0 or theta_label > 179:
                    #     # print('id problems,问题出现在该图片中:%s' % (i, img_fullname))
                    #     print('出问题的longside形式数据:[%.16f, %.16f, %.16f, %.16f, %.1f]' % (
                    #         c_x, c_y, longside, shortside, theta_longside))
                    # # print("theta_label",theta_label)
                    bbox = bbox.reshape(8,)

                    outline = name_id + ' ' + ' '.join(list(map(str, bbox)))
                    f_out.write(outline + '\n')  # 写入txt文件中并加上换行符号 \n
            #
            # points=np.int32(points[:,np.newaxis,:])

        #     img= cv2.drawContours(img, [points], -1, (0, 255, 0), 5)
        # cv2.imwrite("save.jpg",img)
        # cv2.resize(img,(1920,1080))
        # cv2.imshow("img",img)
        # cv2.waitKey(0)


def drawLongsideFormatimg(imgpath, txtpath, yoloimg_i,dict_label):
    """
    根据labels绘制边框(label_format:classid, x_c_normalized, y_c_normalized, longside_normalized, shortside_normalized, Θ)
    :param imgpath: the path of images
    :param txtpath: the path of txt in longside format
    :param dstpath: the path of image_drawed
    :param extractclassname: the category you selected
    """

    # 设置画框的颜色    colors = [[178, 63, 143], [25, 184, 176], [238, 152, 129],....,[235, 137, 120]]随机设置RGB颜色

    filelist = os.listdir(txtpath)

    for fullname in filelist:  # fullname='/.../P000?.txt'

        fullname_path = os.path.join(txtpath, fullname)
        objects = parse_longsideformat(fullname_path)
        '''
        objects[i] = [classid, x_c_normalized, y_c_normalized, longside_normalized, shortside_normalized, theta]
        '''
        img_name = fullname.split(".")[0] + '.jpg'
        img_fullpath = os.path.join(imgpath, img_name)
        img = Image.open(img_fullpath)  # 图像被打开但未被读取
        img_w, img_h = img.size
        scale_max=max(img_w, img_h)

        # print(" img_w, img_h", img_w, img_h)
        img = cv2.imread(img_fullpath)  # 读取图像像素
        for i, obj in enumerate(objects):
            # colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(label_name))]
            # obj = [classid, x_c_normalized, y_c_normalized, longside_normalized, shortside_normalized, float:0-179]

            class_index = obj[0]
            # rect=[(x_c,y_c),(w,h),Θ] Θ:flaot[0-179]  -> (-180,0)
            obj[1]=obj[1]*img_w
            obj[2] = obj[2] * img_h
            obj[3]=obj[3]*img_w
            obj[4] = obj[4] * img_h
            obj[5]=obj[5]*img_w
            obj[6] = obj[6] * img_h
            obj[7]=obj[7]*img_w
            obj[8] = obj[8] * img_h


            poly = np.array([(obj[1],obj[2]),(obj[3],obj[4]),(obj[5],obj[6]),(obj[7],obj[8])])


            # 四点坐标反归一化 取整
            # poly[:, 0] = poly[:, 0] * img_w
            # poly[:, 1] = poly[:, 1] * img_w
            poly[:, 0] = poly[:, 0]
            poly[:, 1] = poly[:, 1]
            # poly[:, 0] = poly[:, 0]
            # poly[:, 1] = poly[:, 1]
            # poly = np.int0(poly)
            poly = np.int32(poly)
            #print("poly",poly)
            # 画出来
            cv2.drawContours(image=img,
                             contours=[poly],
                             contourIdx=-1,
                             color=(0,255,0),
                             thickness=5)
            name=get_dict_key(dict_label,str(class_index))
            #print("dict_label[str(class_index)]",get_dict_key(dict_label,str(class_index)))
            cv2.putText(img, name, (int((poly[0][0]+poly[1][0])/2), int((poly[0][1]+poly[1][1])/2)), 0, 2, [225, 255, 255], thickness=3, lineType=cv2.LINE_AA)
        if not os.path.exists(yoloimg_i):  # 判断是否存在文件夹如果不存在则创建为文件夹
            os.makedirs(yoloimg_i)
        img_save = os.path.join(yoloimg_i, img_name)
        cv2.imwrite(img_save, img)

def get_dict_key(dic,value):
    keys=list(dic.keys())
    values=list(dic.values())
    idx=values.index(value)
    key=keys[idx]
    return key
def longsideformat2cvminAreaRect(x_c, y_c, longside, shortside, theta_longside):
    '''
    trans longside format(x_c, y_c, longside, shortside, θ) to minAreaRect(x_c, y_c, width, height, θ)
    两者区别为:
            当opencv表示法中width为最长边时（包括正方形的情况），则两种表示方法一致
            当opencv表示法中width不为最长边 ，则最长边表示法的角度要在opencv的Θ基础上-90度
    @param x_c: center_x
    @param y_c: center_y
    @param longside: 最长边
    @param shortside: 最短边
    @param theta_longside: 最长边和x轴逆时针旋转的夹角，逆时针方向角度为负 [-180, 0)
    @return: ((x_c, y_c),(width, height),Θ)
            x_c: center_x
            y_c: center_y
            width: x轴逆时针旋转碰到的第一条边最长边
            height: 与width不同的边
            theta: x轴逆时针旋转与width的夹角，由于原点位于图像的左上角，逆时针旋转角度为负 [-90, 0)
    '''
    if (theta_longside >= -180 and theta_longside < -90):  # width is not the longest side
        width = shortside
        height = longside
        theta = theta_longside + 90
    else:
        width = longside
        height = shortside
        theta = theta_longside

    if theta < -90 or theta >= 0:
        print('当前θ=%.1f，超出opencv的θ定义范围[-90, 0)' % theta)

    return ((x_c, y_c), (width, height), theta)


def GetFileFromThisRootDir(dir, ext=None):
    allfiles = []
    needExtFilter = (ext != None)
    for root, dirs, files in os.walk(dir):
        for filespath in files:
            filepath = os.path.join(root, filespath)
            extension = os.path.splitext(filepath)[1][1:]
            if needExtFilter and extension in ext:
                allfiles.append(filepath)
            elif not needExtFilter:
                allfiles.append(filepath)
    return allfiles


def parse_longsideformat(filename):  # filename=??.txt
    """
        parse the longsideformat ground truth in the format:
        objects[i] : [classid, x_c, y_c, longside, shortside, theta]
    """
    objects = []
    f = []
    if (sys.version_info >= (3, 5)):
        fd = open(filename, 'r')
        f = fd
    elif (sys.version_info >= 2.7):
        fd = codecs.open(filename, 'r')
        f = fd
    # count = 0
    while True:
        line = f.readline()
        if line:
            splitlines = line.strip().split(' ')
            object_struct = {}
            ### clear the wrong name after check all the data
            # if (len(splitlines) >= 9) and (splitlines[8] in classname):
            if (len(splitlines) < 9) or (len(splitlines) > 9):
                print('labels长度不为9,出现错误,与预定形式不符')
                continue

            object_struct = [int(splitlines[0]), float(splitlines[1]),
                             float(splitlines[2]), float(splitlines[3]),
                             float(splitlines[4]), float(splitlines[5]),
                             float(splitlines[6]), float(splitlines[7]),
                             float(splitlines[8])
                             ]
            objects.append(object_struct)
        else:
            break
    return objects
file = ["./Annotations"]
img = ["./imgs"]
txtpath = ["./labels_poly"]
label_name = ['seg_jueyuanzi_01', 'seg_fangzhenchui_sh', 'seg_daodixian_01', 'fushusheshi_01', 'fushusheshi_03',
              'ganta_02', 'gkxfw', 'TaDiao', 'YanHuo', 'zhongjianjian']
yolo_img = ["./yolo_imgs"]
dict_label = {'seg_jueyuanzi_01': '0', 'seg_fangzhenchui_sh': '1', 'seg_daodixian_01': '2',
              'fushusheshi_01': '3', 'fushusheshi_03': '4', 'ganta_02': '5',
              'gkxfw': '6', 'TaDiao': '7', 'YanHuo': '8', 'zhongjianjian': '9'}

for file_i, img_i, txt_i, yoloimg_i in zip(file, img, txtpath, yolo_img):
    # class_id=dict_label[label_i]
    get_minrec(file_i, img_i, txt_i, label_name, dict_label)
    drawLongsideFormatimg(img_i, txt_i, yoloimg_i,dict_label)