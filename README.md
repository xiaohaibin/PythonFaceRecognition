>最近在研究一个人脸识别开源框架[face_recognition](https://github.com/ageitgey/face_recognition)，编译需要依赖 dlib、boost、cmake 等相关环境，在编译的时候，踩了一大堆坑，网上资料大都不是很全面再加上 Windows 环境下去配置这些本身就比 Liunx 或 Mac下配置要相对麻烦一些。如果你是准备在 Windows 环境下编译，就做好准备踩坑吧~~~

>系统环境
>* Windows 7
>* Python 2.7.14
>* VS2015

### 安装步骤

**1\. 首先[https://github.com/davisking/dlib](https://github.com/davisking/dlib)  下载整个zip**
``` 
git clone https://github.com/davisking/dlib
``` 


**2\. 前置的一些python库要安装, scipy, numpy+mkl  这两个用pip安装可能会蛋疼, [https://www.lfd.uci.edu/~gohlke/pythonlibs/](https://www.lfd.uci.edu/~gohlke/pythonlibs/) 到这里面找对应版本的wheel, 然后用easy_install就KO了**

**3\. 安装Boost  [https://sourceforge.net/projects/boost/files/boost-binaries](https://sourceforge.net/projects/boost/files/boost-binaries)  下载boost-binaries, 最新的,直接点击exe程序等待安装完毕， 正常的话安装的目录是 X:\local\boost_1_XX_X (保持版本名一致, 也就是XX_X别改)**

**4\. 这一步也貌似可以不用  系统变量加上 VS2015的位置  新建一个**
``` 
VS140COMNTOOLS 值  X:\Program Files (x86)\Microsoft Visual Studio 14.0\Common7\Tools\
``` 

**5\. 去到Boost下面, 双击一下 bootstrap.bat , 运行OK之后boost_1_66_0\tools\build文件夹下找到以下两个文件**

![](https://upload-images.jianshu.io/upload_images/1956769-d82690d6170d6ff7.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

>然后开启一个命令行，定位到这个文件夹，运行命令：
``` 
b2 install
``` 
>运行完之后再执行下面命令：
``` 
 b2 -a --with-python address-model=64 toolset=msvc runtime-link=static 
``` 
>成功后能找到 stage 文件夹

**6\.在系统环境配置好下面两个环境变量**

``` 
BOOST_ROOT=C:\local\boost_X_XX_X

BOOST_LIBRARYDIR=C:\local\boost_X_XX_X\stage\lib 

``` 

>最后执行以下命令：
``` 
python setup.py install --yes USE_AVX_INSTRUCTIONS --yes DLIB_USE_CUDA
``` 
>CUDA这个, 主要是用在有显卡的机器学习, 如果没有可以不加
完成! 打开python shell 试一下 import dlib 没问题就可以
``` 
pip install face_recognition
``` 

>安装成功之后，我们可以在python中正常 import face_recognition 了

![](https://upload-images.jianshu.io/upload_images/1956769-02825ff33bc647df.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


#### 编写人脸检测程序
>此demo主要展示了识别指定图片中人脸的特征数据，下面就是人脸的八个特征，我们就是要获取特征数据

        'chin',
        'left_eyebrow',
        'right_eyebrow',
        'nose_bridge',
        'nose_tip',
        'left_eye',
        'right_eye',
        'top_lip',
        'bottom_lip'

**人脸检测代码**
``` 
# -*- coding: utf-8 -*-
# 自动识别人脸特征
# filename : find_facial_features_in_picture.py

# 导入pil模块 ，可用命令安装 apt-get install python-Imaging
from PIL import Image, ImageDraw
# 导入face_recogntion模块，可用命令安装 pip install face_recognition
import face_recognition

# 将jpg文件加载到numpy 数组中
image = face_recognition.load_image_file("mayun.jpg")

#查找图像中所有面部的所有面部特征
face_landmarks_list = face_recognition.face_landmarks(image)

print("I found {} face(s) in this photograph.".format(len(face_landmarks_list)))

for face_landmarks in face_landmarks_list:

   #打印此图像中每个面部特征的位置
    facial_features = [
        'chin',
        'left_eyebrow',
        'right_eyebrow',
        'nose_bridge',
        'nose_tip',
        'left_eye',
        'right_eye',
        'top_lip',
        'bottom_lip'
    ]

    for facial_feature in facial_features:
        print("The {} in this face has the following points: {}".format(facial_feature, face_landmarks[facial_feature]))

   #让我们在图像中描绘出每个人脸特征！
    pil_image = Image.fromarray(image)
    d = ImageDraw.Draw(pil_image)

    for facial_feature in facial_features:
        d.line(face_landmarks[facial_feature], width=5)

    pil_image.show() 
``` 
>运行结果：
自动识别图片中的人脸，并且识别它的特征

**原图：**

![原图](https://upload-images.jianshu.io/upload_images/1956769-f7c99ef8d5bf38e2.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

**运行效果：**
![运行结果](https://upload-images.jianshu.io/upload_images/1956769-eb8d26c533952d5e.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

#### 编写人脸识别程序
>注意：这里使用了 python-opencv，一定要配置好了opencv才能运行成功。
opencv选择跟自己python版本相匹配的版本，可以在这个网站（https://www.lfd.uci.edu/~gohlke/pythonlibs/）下载opencv_python-2.4.13.5-cp27-cp27m-win_amd64 .whl(我的python版本是2.7所以选择该版本安装)，安装完成之后，打开 cmd 输入 import cv2 没有提示任何错误说明安装成功。

![opencv安装成功](https://upload-images.jianshu.io/upload_images/1956769-d7edc11d8d8ef15f.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


``` 
# -*- coding: utf-8 -*-
#

# 检测人脸
import face_recognition
import cv2

# 读取图片并识别人脸
img = face_recognition.load_image_file("mayun.jpeg")
face_locations = face_recognition.face_locations(img)
print face_locations

# 调用opencv函数显示图片
img = cv2.imread("mayun.jpeg")
cv2.namedWindow("原图")
cv2.imshow("原图", img)

# 遍历每个人脸，并标注
faceNum = len(face_locations)
for i in range(0, faceNum):
    top =  face_locations[i][0]
    right =  face_locations[i][1]
    bottom = face_locations[i][2]
    left = face_locations[i][3]
    
    start = (left, top)
    end = (right, bottom)
    
    color = (55,255,155)
    thickness = 3
    cv2.rectangle(img, start, end, color, thickness)

# 显示识别结果
cv2.namedWindow("人脸识别")
cv2.imshow("人脸识别", img)

cv2.waitKey(0)
cv2.destroyAllWindows()
``` 
>运行结果： 
程序会读取当前目录下指定的图片，然后识别其中的人脸，并标注每个人脸。 
![人脸识别结果](https://upload-images.jianshu.io/upload_images/1956769-5415a500775a2808.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

#### 摄像头实时识别人脸
>此处因为公司台式机没有摄像头，所以用的Mac上运行这个demo，配置都是差不多的，环境配置好，运行下面代码即可
``` 
# -*- coding: utf-8 -*-
# 摄像头实时识别人脸
import face_recognition
import cv2

video_capture = cv2.VideoCapture(0)
# 加载本地图片
xhb_img = face_recognition.load_image_file("xhb.jpg")
xhb_face_encoding = face_recognition.face_encodings(xhb_img)[0]

# 初始化变量
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    ret, frame = video_capture.read()
    
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    
    if process_this_frame:
        face_locations = face_recognition.face_locations(small_frame)
        face_encodings = face_recognition.face_encodings(small_frame, face_locations)
        
        face_names = []
        for face_encoding in face_encodings:
            # 可以自定义设置识别阈值（tolerance）此处设置为0.5，默认为0.6，太小可能识别不出来，太大可能造成识别混淆
            match = face_recognition.compare_faces([xhb_face_encoding], face_encoding,tolerance=0.5)
            
            if match[0]:
                name = "xiaohaibin"
            else:
                name = "unknown"
            
            face_names.append(name)

    process_this_frame = not process_this_frame
    
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255),  2)
        
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left+6, bottom-6), font, 1.0, (255, 255, 255), 1)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
``` 
>运行结果展示
![实时人脸识别](https://upload-images.jianshu.io/upload_images/1956769-0b6278cba8513a8e.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
 
### 相关资料
*   [ face_recognition](https://github.com/ageitgey/face_recognition)
*   [OpenCv ](https://opencv.org/)
*   [Boost C++ Libraries](https://sourceforge.net/projects/boost/files/)
*  [人工智能之Python人脸识别技术--face_recognition模块 ](https://blog.csdn.net/qq_31673689/article/details/79370412)
*  [应用一个基于Python的开源人脸识别库face_recognition](https://blog.csdn.net/hongbin_xu/article/details/76284134)

### 踩坑

#### 1.解决 cl.exe 找不到的问题

>在装VS2015时，默认是不安装C++，你需要重新运行setup ，然后选择modify,选择 language 下的C++，然后开始安装，就可以解决问题了
–来自[http://stackoverflow.com/](http://stackoverflow.com/)

![VS2015配置.png](https://upload-images.jianshu.io/upload_images/1956769-79c671dd67f62f40.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

#### 2.解决执行 pip install face_recognition 出错，报SSL Error

![SSL Error](https://upload-images.jianshu.io/upload_images/1956769-07dc6eee09aa8160.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

**方案一(推荐)：在pip安装所在文件夹路径下，创造python文件(.py)**

``` 
import os  

ini="""[global] 
index-url = https://pypi.doubanio.com/simple/ 
[install] 
trusted-host=pypi.doubanio.com 
"""  
pippath=os.environ["USERPROFILE"]+"\\pip\\"  

if not os.path.exists(pippath):  
    os.mkdir(pippath)  

with open(pippath+"pip.ini","w+") as f:  
    f.write(ini)  
``` 

>在cmd上运行这个.py文件即可
之后再用pip install安装指令下载速度会非常快

**方案二：修改加大超时时间**

``` 
pip --default-timeout=100 install -U pip
``` 
>再执行下面指令进行安装
``` 
pip --default-timeout=100 install -U face_recognition
``` 
#### 3.Python脚本报错AttributeError: ‘module’ object has no attribute’xxx’解决方法

>最近在编写Python脚本过程中遇到一个问题比较奇怪：Python脚本完全正常没问题，但执行总报错"AttributeError: 'module' object has no attribute 'xxx'"。这其实是.pyc文件存在问题。

**问题定位：**

>查看import库的源文件，发现源文件存在且没有错误，同时存在源文件的.pyc文件

**问题解决方法：**

>1. 命名py脚本时，不要与python预留字，模块名等相同
>2. 删除该库的.pyc文件（因为py脚本每次运行时均会生成.pyc文件；在已经生成.pyc文件的情况下，若代码不更新，运行时依旧会走pyc，所以要删除.pyc文件），重新运行代码；或者找一个可以运行代码的环境，拷贝替换当前机器的.pyc文件即可。
