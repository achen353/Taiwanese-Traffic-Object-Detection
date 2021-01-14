<h1 align="center">Object Detection on Taiwanese Traffic using YOLOv4 Tiny</h1>

<div align="center">
    <strong>Exploration of YOLOv4 Tiny on custom Taiwanese traffic dataset</strong>
</div>

<div align="center">
    Trained and tested AlexeyAB's Darknet YOLOv4 Tiny on Nvidia Tesla P100 GPU
</div>

<br/>

<div align="center">
    <!-- Framework  -->
    <img src="https://img.shields.io/badge/framework-Darknet-blue"/>
    <!-- Model author -->
    <img src="https://img.shields.io/badge/model-by%20AlexeyAB-blue"/>
    <!-- Last commit -->
    <img src="https://img.shields.io/github/last-commit/achen353/Taiwan-Traffic-Object-Detection?style=flat-square"/>
    <!-- Stars -->
    <img src="https://img.shields.io/github/stars/achen353/Taiwan-Traffic-Object-Detection?style=flat-square"/>
    <!-- Forks -->
    <img src="https://img.shields.io/github/forks/achen353/Taiwan-Traffic-Object-Detection?style=flat-square"/>
    <!-- Issues -->
    <img src="https://img.shields.io/github/issues/achen353/Taiwan-Traffic-Object-Detection?style=flat-square"/>
</div>

<br/>

<div align="center">
    <img src="https://github.com/achen353/Taiwan-Traffic-Object-Detection/blob/master/readme_assets/prediction_night.gif" width="80%"/>
</div>

## Table of Contents
- [About](#about)
- [Framework and Model](#framework-and-model)
- [Dataset](#dataset)
- [Results](#results)
- [Setup](#setup) 
- [How to Run](#how-to-run)
- [Credits](#credits)

## About
In this project, we trained and fine-tuned the [YOLOv4 Tiny model](https://arxiv.org/abs/2004.10934) on a custom 
dataset of Taiwanese traffic provided by the
[Embedded Deep Learning Object Detection Model Compression Competition for Traffic in Asian Countries](https://aidea-web.tw/topic/35e0ddb9-d54b-40b7-b445-67d627890454?focus=intro)
as training data. 

Released in April 2020 by AlexeyAB and his team, respectively, YOLOv4 became one of the fastest object 
recognition system on the market. It was even deployed as [real-time object detection systems](https://www.taiwannews.com.tw/en/news/3957400) 
as a solution to traffic flow problems in Taiwanese cities such as Taoyuan City and in Hsinchu City.

The project explores the capability of YOLOv4 Tiny, the lightweight version of YOLOv4 developed by the same research 
team. Moreover, as there has been little publicly available object detection datasets on Taiwanese and Asian traffic
(where large numbers of scooters and bicycles are present compared to most Western countries),
we took the chance to utilize the dataset provided by Embedded Deep Learning Object Detection Model Compression 
Competition for Traffic in Asian Countries. 

We achieved an 87.2% mAP@0.5 at about 18-23 average FPS with Nvidia Tesla P100 GPU.

DISCLAIMER: This project is an application, not an implementation/modification, of YOLOv4 Tiny. Most of the code is from
the original YOLOv4 repository.

## Framework and Model
Both YOLOv4 and YOLOv4 Tiny are implemented by AlexeyAB and his team, based on the
[forked repository](https://github.com/AlexeyAB/darknet) from Joseph Redmon's 
[original work](https://github.com/pjreddie/darknet) on YOLOv1, YOLOv2, and YOLOv3. While there are many implementations
of the models in other frameworks such as Tensorflow and PyTorch, we decided to familiarize ourselves with the original
Darknet framework.

Visit the original Darknet [repo](https://github.com/AlexeyAB/darknet) to learn more about the models themselves as well
as implementations in other frameworks.  

## Dataset
The dataset consists of `89002` images of size `1920*1080`. There are `4` annotated classes: vehicle, scooter,
pedestrian, bicycle. All the images are provided with annotations as training data for the participants of the 
Embedded Deep Learning Object Detection Model Compression Competition for Traffic in Asian Countries. We used `80%` of 
it as training data and `20%` as validation (and test) data. 

Mandated by the host of the competition, the data is kept confidential. However, we do provide the weights trained and
demonstrations of the model's performance as provided below.

## Results
Setting the model resolution `1280*704`, we were able to achieve an 87.2% mAP@0.5 at 18-23 FPS on average.

<div align="center">
    <img src="https://github.com/achen353/Taiwan-Traffic-Object-Detection/blob/master/chart_yolov4-tiny-obj.png" width="50%"/>
</div>

Below are examples of our model making inferences:

Sunny Day (Avg FPS: 23.0)                     |  Sunny Night (Avg FPS: 19.7)
:---------------------------------------------:|:------------------------------------------------:
![](./readme_assets/prediction_day.gif)        |  ![](./readme_assets/prediction_night.gif)

Rainy Day (Avg FPS: 20.2)                     |  Rainy Night (Avg FPS: 20.3)
:---------------------------------------------:|:------------------------------------------------:
![](./readme_assets/prediction_day_rain.gif)   |  ![](./readme_assets/prediction_night_rain.gif) 

Check out `predictions` folders if the `.gif` above are loading slow.

## Setup
1. Open your terminal, `cd` into where you'd like to clone this project, and clone the project:
```
$ git clone https://github.com/achen353/Taiwan-Traffic-Object-Detection.git
```
2. Follow the steps in this [article](https://robocademy.com/2020/05/01/a-gentle-introduction-to-yolo-v4-for-object-detection-in-ubuntu-20-04/#Installing_YOLO_Prerequisites). 
   It has very detailed instructions of setting up the environment for training YOLO models.
   
Note: Make sure your CUDA, cuDNN, CUDA driver and your hardware are compatible. Check compatibility [here](https://docs.nvidia.com/deeplearning/cudnn/support-matrix/index.html).

It is highly suggested that you run YOLO with a GPU. If you don't have a GPU, you can use the following free resources 
Google provides:

- In the US: You can run on Google Colab with free Tesla K80 GPU. Check out these tutorials:
  - [YOLOv4 on Google Colab: Train your Custom Dataset (Traffic signs) with ease](https://towardsdatascience.com/yolov4-in-google-colab-train-your-custom-dataset-traffic-signs-with-ease-3243ca91c81d)
  - [How to Train YOLOv4 on a Custom Dataset](https://blog.roboflow.com/training-yolov4-on-a-custom-dataset/)
- Outside the US: If you've never used Google Cloud Platform, you can start a free-trial with $300 credits. Some 
  articles are helpful in setting up your own deep learning VM instance:
  - [How to run Deep Learning models on Google Cloud Platform in 6 steps?](https://medium.com/google-cloud/how-to-run-deep-learning-models-on-google-cloud-platform-in-6-steps-4950a57acfa5)
  - [Setting up Jupyter Lab Instance on Google Cloud Platform](https://medium.com/analytics-vidhya/setting-up-jupyter-lab-instance-on-google-cloud-platform-3a7acaa732b7)
  - [Installing CUDA on Google Cloud Platform in 10 minutes](https://towardsdatascience.com/installing-cuda-on-google-cloud-platform-in-10-minutes-9525d874c8c1)
  - [Setting up Chrome Remote Desktop for Linux on Compute Engine](https://cloud.google.com/solutions/chrome-desktop-remote-on-compute-engine)    

If you're training on a remote server via SSH session, this [article](https://www.tecmint.com/screen-command-examples-to-manage-linux-terminals/)
will help you in keeping the terminal session alive even if you disconnect the SSH session.

## How to Run

### To train your own model on a custom dataset
The [original repository](https://github.com/AlexeyAB/darknet#how-to-train-to-detect-your-custom-objects) has very 
detailed steps for training on a custom dataset

### To inference using the pre-trained model provided by this project
1. Get any `.avi`/`.mp4` video file (preferably not more than 1920*1080 to avoid bottlenecks in CPU performance) and 
   place them in where you prefer (we had `sample_videos` folder for this).
2. Run `make` if you haven't and run (assume the file name is `test.mp4`):
```
$ ./darknet detector demo data/obj.data cfg/yolov4-tiny-obj.cfg yolov4-tiny-obj_best.weights sample_videos/test.mp4 -out_filename predictions/test.mp4
```
We have `prediction` folder made to keep the prediction outputs. If you don't want the video to pop up as the model 
makes inferences, add `-dont_show` flag in your command.

## Credits
- AlexeyAB's Darknet YOLOv4 [repository](https://github.com/AlexeyAB/darknet) 
- [Embedded Deep Learning Object Detection Model Compression Competition for Traffic in Asian Countries](https://aidea-web.tw/topic/35e0ddb9-d54b-40b7-b445-67d627890454?focus=intro)


