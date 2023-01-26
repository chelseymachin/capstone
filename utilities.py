import re
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plotter
import pandas
import random
import numpy
from transformers.file_utils import is_tf_available, is_torch_available, is_torch_tpu_available
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import BertTokenizerFast, BertForSequenceClassification

# initialize objects for the stemmer and lemmatizer from nltk
stemmer = PorterStemmer()
word_net_lemmatizer = nltk.stem.WordNetLemmatizer()

# initialize a dictionary of english stop words from nltk
nltk.download('stopwords')
stop_words = stopwords.words('english')
stopwords_dictionary = Counter(stop_words)

# function that cleans up text by removing URLs, extra whitespace, special characters, etc
def text_cleaner(input):
    input = str(input).replace(r'http[\w:/\.]+', ' ')
    input = str(input).replace(r'[^\.\w\s]', ' ')
    input = str(input).replace(r'[^a-zA-Z]', ' ')
    input = str(input).replace(r'\s\s+', ' ')
    input = input.lower().strip()

    return input

# function that pre-processes text by removing stop words and getting the 'lemma' for the words
def preprocess(input):
    input = text_cleaner(input)
    wordlist = re.sub(r'[^\w\s]', '', input).split()
    input = ' '.join([word_net_lemmatizer.lemmatize(input) for input in wordlist if input not in stopwords_dictionary])
    
    return input

# generates a wordcloud for the input with standardized settings
def wordcloud_generator(input):
    wordcloud = WordCloud(background_color='pink', width=300, height=300)
    text_on_cloud = wordcloud.generate(input)
    plotter.figure(figsize=(6, 6))
    plotter.imshow(text_on_cloud)
    plotter.axis('off')

# generate a histogram showing 12 most occurring n combinations of words for the input given with standardized settings
def plot_most_occurring_combinations(corpus, title, ylabel, xlabel='# of Times Used', n=2,):
    top_combinations = (pandas.Series(nltk.ngrams(corpus.split(), n)).value_counts())[:12]
    top_combinations.sort_values().plot.barh(color='pink', width=.8, figsize=(8, 8))
    plotter.title(title)
    plotter.ylabel(ylabel)
    plotter.xlabel(xlabel)
    plotter.show()

# function to set seed; makes training efforts reproducible
def seed(int):
    random.seed(int)
    numpy.random.seed(int)

    if is_torch_available():
        torch.manual_seed(int)
        torch.cuda.manual_seed_all(int)

# function that creates arrays for texts and labels to be split into a train/test split stack 
def prep_to_train(data, test_size=0.2):
    texts = []
    labels = []
    
    for i in range(len(data)):
        text = data['text'].iloc[i]
        label = data['label'].iloc[i]
        text = data['title'].iloc[i] + ' - ' + text
        if text and label in [0, 1]:
            texts.append(text)
            labels.append(label)

    return train_test_split(texts, labels, test_size=test_size)

# function to determine accuracy of a split; utilizes accuracy_score function from scikitlearn
def determine_metrics(input):
    labels = input.label_ids
    predictions = input.predictions.argmax(-1)
    accuracy = accuracy_score(labels, predictions)
    return {
        'accuracy': accuracy
    }

# function to get a prediction from the final, fine-tuned BERT model; accepts text input and initializes the model and tokenizer from their file path
# input is then run through tokenizer and passed to the model
# probabilities list is generated and highest/first is returned after being converted to the proper label
def predict(input):
    model = BertForSequenceClassification.from_pretrained('the-final-bert-base-model', num_labels=2)
    max_length = 512
    tokenizer = BertTokenizerFast.from_pretrained('the-final-bert-base-model', do_lower_case=True)

    inputs = tokenizer(input, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    outputs = model(**inputs)
    probabilities = outputs[0].softmax(1)
    labels = {
        0: "true news",
        1: "fake news"
    }

    return labels[int(probabilities.argmax())]