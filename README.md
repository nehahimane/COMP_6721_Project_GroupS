# COMP_6721_Project_GroupS
COMP_6721_Applied_AI_Project

## Group members

1) Neha Himane
2) Yash Bhavsar
3) Hardik Amareliya
4) Jeet Ambaliya

Project : Facial Expression Recognition

This project represents the image classification problem and mapping of different facial expression images to their corresponding emotional classes such as happy, sad, disappointed, neutral, angry, surprise and interested. Models used are AlexNet, VGG19, and MobileNet along with AffectNetHQ, Fer_Custom and Black&White datasets from Kaggle. We trained 11 models in total in total where 2 of them are transfer learning. AlexNet with FerCustom and BW Dataset are utilised for transfer learning. Data is standardized in data preprocessing, features are extracted using cross entropy loss as criterion and SGD as optimizer and the comparison results across different model and datasets have been achieved and plotted.



Below list of libraries need to be installed. Use !pip install <name> in case of missing library:

import os
import torch
import torchvision
import tarfile
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchvision.datasets.utils import download_url
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.data import random_split
from torchvision.utils import make_grid
import matplotlib
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler
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
from matplotlib import image
from matplotlib import pyplot
import time
import torchvision.datasets as datasets
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


- Your source code package in PyTorch
- Description on how to obtain the Dataset from an available download link
Original dataset links of kaggel:
Dataset 1:
Dataset 2:
Dataset 3:

Google drive links of Dataset we used:
Dataset 1:
Dataset 2:
Dataset 3:
