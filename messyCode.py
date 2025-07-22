import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

nltk.download('punkt')
nltk.download('stopwords')

df = pd.read_csv('data.csv')


def clean(x):
    words = word_tokenize(x.lower())
    w = []
    for i in words:
        if i not in stopwords.words('english') and i not in string.punctuation:
            w.append(i)
    return ' '.join(w)


df['text'] = df['Text'].apply(clean)

print(df.head())
