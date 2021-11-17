# Rotation-Detect-yolov5_poly
本项目主要基于yolov5算法的旋转目标检测，主要借鉴项目https://github.com/hukaixuan19970627/YOLOv5_DOTA_OBB

https://github.com/acai66/yolov5_rotation

本项目主要对这两个项目中存在的问题进行改进，主要改进点为针对目标出现在图像边缘位置时，图像预处理会出现label偏移现象，以及对边缘目标没有目标截断操作等预处理方式进行改进。主要改进方式为抛弃原始的yolo形式的预处理方式（centetx,centery,w,h），改成直接对
旋转矩形框进行预处理，新的label数据类型为（x1,y1,x2,y2,x3,y3,x4,y4）

## Installation (Linux Recommend, Windows not Recommend)
```
conda create -n rotation_yolo_poly python=3.6

conda activate rotation_yolo_poly

conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch

cd Rotation-Detect-yolov5_poly

pip install -r requirements
```

注意我的环境中torch版本是1.9.0，理论上1.7以上版本torch均可运行

## Usage Example
### 1.制作数据集
通过labelme软件标注的rectangle或者poly均可，标注完成的数据集目录为：

|_train

----|_imgs:

----------|1.jpg

----------|2.jpg

----|_Annotations:

----------|1.json

----------|2.json

----|_min_poly.py

|_test

----|_imgs:

----------|1.jpg

----------|2.jpg

----|_Annotations:

----------|1.json

----------|2.json

----|_min_poly.py
labelme标注的实例如下图，标注成poly类型或者rec类型均可
![image](https://user-images.githubusercontent.com/28287748/142178999-246c9059-2507-42cb-8e1c-c17c2567e88a.png)

### 2.数据预处理
```
python min_poly.py
```

直接运行min_poly.py文件进行预处理（注意：环境为之前创建的conda虚拟环境）,修改min_poly.py中的label_name为自己的类别名称，修改min_poly.py中的115行的name_id为自己的标注文件中标注为多边形poly的类别。（默认已完成训练集和测试集分割）

### 3.生成训练集列表

```
python labeldir.py
```

### 4.开始训练

#### （1）修改/data路径下的DOTA——ROTATED.yaml文件，将其中的train,val,test，替换成自己的路径，将nc改成自己的类别数，name改成自己的类别名称

#### （2）修改/utils文件夹中的datasets.py文件中的535行，将在其中的'img'和'labels_poly'改成自己的数据集中的图片文件夹名和label文件夹名称。

#### （3）运行train.py,对于单卡可直接运行:

`python train.py  --batch-size 4 --device 0`

对于多卡训练，运行：

`python -m torch.distributed.launch --nproc_per_node 4 train.py  --device 0,1,2,3`

#### （4）评估训练效果

`python val.py`



