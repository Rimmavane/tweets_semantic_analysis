import nltk
import pandas as pd
import re
import string
nltk.download('stopwords')
from nltk.corpus import stopwords

url_regex = re.compile(r'^(?:(?:https?|ftp)://)(?:\S+(?::\S*)?@)?(?:(?:[1-9]\d?|1\d\d|2[01]\d|22[0-3])'
                       r'(?:\.(?:1?\d{1,2}|2[0-4]\d|25[0-5])){2}(?:\.(?:[1-9]\d?|1\d\d|2[0-4]\d|25[0-4]))'
                       r'|(?:(?:[a-z\u00a1-\uffff0-9]+-?)*[a-z\u00a1-\uffff0-9]+)(?:\.(?:[a-z\u00a1-\uffff0-9]+-?)'
                       r'*[a-z\u00a1-\uffff0-9]+)*(?:\.(?:[a-z\u00a1-\uffff]{2,})))(?::\d{2,5})?(?:/[^\s]*)?$',
                       re.IGNORECASE)                   # detects (most) URLs

nickname_regex = re.compile(r'@[A-Za-z0-9]+')           # detects nicknames
punc_and_digits = string.punctuation + string.digits    # detects punctuation and digits


def load_data(data_path):
    """
    Load data as Pandas Dataframe if it is csv, or line by line if otherwise.
    """
    if '.csv' in data_path:
        return pd.read_csv(data_path, header=None)
    else:
        with open(data_path, 'r') as handle:
            data = [seq.strip() for seq in handle]
            return data


def clean_sentence(text):
    """
    Perform basic string cleaning - removes URLs, punctuation, digits, stopwords - and return a string.
    """
    # clean sentence if a string is given
    if type(text) is str:
        text = text.lower()
        text = re.sub(url_regex, "", text)                                          # remove URLS
        text = text.translate(str.maketrans('', '', punc_and_digits))               # remove punctuation and digits
        text = text.split()                                                         # split by whitespaces
        text = [word for word in text if word not in stopwords.words("english")]    # remove stopwords
        return ' '.join(text)                                                       # return as single string
