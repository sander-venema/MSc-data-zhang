import torch
import os
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image

from archs.model_wass import Generator, Discriminator
from utils.data_stuff import GenerationDataset

from tqdm import tqdm

import argparse

parser = argparse.ArgumentParser(description='Store training settings')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--learning_rate', type=float, default=0.0002, help='Learning rate')
parser.add_argument('--loss', type=int, default=0, help='Loss function; 0: Wasserstein,')

args = parser.parse_args()

BATCH_SIZE = args.batch_size
LEARNING_RATE = args.learning_rate

#TODO: extend to cover all settings