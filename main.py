# coding=utf8
import pandas
import numpy
import seaborn
import re
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
import random
import utilities

# read in the training file into a pandas data frame
data = pandas.read_csv('train.csv')

# drop the unneeded ID column from the data
data = data.drop(['id'], axis=1)

# replace any null values in the text field cells with 'None' instead
data.loc[data['title'].isnull(), 'title'] = 'None'
data.loc[data['text'].isnull(), 'text'] = 'None'

# apply the pre-processing utility function from the utilities file to the text cells in the data
data['text'] = data.text.apply(utilities.preprocess)
data['title'] = data.title.apply(utilities.preprocess)

# make joined strings of fake and true news data for visualization purposes
true_news = ' '.join(data[data['label']==0]['text'])
fake_news = ' '.join(data[data['label']==1]['text'])
