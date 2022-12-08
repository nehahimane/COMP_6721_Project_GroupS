# COMP_6721_Project_GroupS
COMP_6721_Applied_AI_Project

## Group members

1) Neha Himane - 40219032
2) Yash Bhavsar - 40219504
3) Hardik Amareliya - 40216854
4) Jeet Ambaliya - 40221712

- Project : Facial Expression Recognition

This project represents the image classification problem and mapping of different facial expression images to their corresponding classes such as happy, sad, disappointed, neutral, angry, surprise and interested. Models used are AlexNet, VGG19, and MobileNet along with AffectNetHQ, Fer_Custom and Black&White datasets from Kaggle. We trained 11 models in total in total where 2 of them were trained using transfer learning. AlexNet with FerCustom and BW Dataset are utilised for transfer learning. Data is standardized in data preprocessing, features are extracted using cross entropy loss as criterion and SGD as optimizer and the comparison results across different model and datasets have been achieved and plotted using various techniques like linegraph, confusion matrix and tsne visualization.


- Below list of libraries need to be installed. Use !pip install <name> in case of missing library:

import os
import torch
import torchvision
import tarfile
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import matplotlib
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import glob
import torch.optim as optim
import cv2
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as td
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import time
import torchvision.datasets as datasets
from torch.utils.data import random_split
from torchvision.utils import make_grid
from torchvision.datasets.utils import download_url
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from matplotlib import image
from matplotlib import pyplot
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix
from ptflops import get_model_complexity_info
%matplotlib inline


- Instruction on how to train/validate your model
For Google colab,
    1) mount the drive
    2) copy the zip file from drive to colab environment
    3) extract the zip file into colab environment
    4) For the training use GPU allocated to Colab Runtime which is provided by Google Colab with CUDA cores. 
    5) run the code

For Saturn Cloud,
    1) Upload the dataset zip to the Saturn Workspace as it gives the Jupyter Notebook
    2) Create the folder for Dataset. 
    3) Extract the Zip file to the Dataset folder. 
    4) Remove the Zip file from workspace runtime as the storage disk is limited in Saturn Cloud.
    5) run the code

All the results are plotted using matplotlib line graphs, confusion matrix and TSNE.


- Instructions on how to run the pre-trained model on the provided sample test dataset
Run Transfer Learning Model:

1) Transfer Learning Models
  1) Load the saved model using below Code:
  
          model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet')
          model.classifier[4] = nn.Linear(4096,1024)
          model.classifier[6] = nn.Linear(1024,3)       # 3 for fer custom dataset and 6 for black and White Dataset
          model.load_state_dict(torch.load(<google drive link for .pt file>))
          device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
          print("Device: {}".format(device))
          model.to(device)
          model.eval()
          
    2) Now use this model for Testing on sample test dataset
2) Without Transfer Learning
  1) Load the saved model using below Code:
  
          #model_name = ['alexnet', 'vgg19', 'mobilenet_v2']
          model = torch.hub.load('pytorch/vision:v0.10.0', <model name>)
          model.load_state_dict(torch.load(<google drive link for .pt file>))
          device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
          print("Device: {}".format(device))
          model.to(device)
          model.eval()
          
  2) Now use this model for Testing on sample test dataset
  
- Your source code package in PyTorch
Link: https://drive.google.com/drive/folders/17Ut-tswXRbrZ7Z_4P_ccC2TyrAfL41Ce?usp=share_link

- Original dataset links of kaggel:
Dataset 1: https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset
Dataset 2: https://www.kaggle.com/datasets/tom99763/affectnethq
Dataset 3: https://www.kaggle.com/datasets/nightfury007/fercustomdataset-3classes 

- Google drive links of datasets we used:
      We have given google drive link for 3 dataset.
Datasets: https://drive.google.com/drive/folders/1eIip-YiNYHQTVFV4N5XdQvJKIRFk2-fv?usp=share_link
