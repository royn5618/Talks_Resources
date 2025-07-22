import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

def clean_text(text, stop_words):
    """
    Cleans input text by:
    - Lowercasing
    - Tokenizing
    - Removing stopwords and punctuation

    Args:
        text (str): Input text string.
        stop_words (set): Set of English stopwords.

    Returns:
        str: Cleaned and space-joined string.
    """
    tokens = word_tokenize(text.lower())
    cleaned = [word for word in tokens if word not in stop_words and word not in string.punctuation]
    return ' '.join(cleaned)

def main():
    try:
        # Load data
        df = pd.read_csv('data.csv')  # Replace with your actual path
        if 'Text' not in df.columns:
            raise KeyError("Column 'Text' not found in CSV.")

        stop_words = set(stopwords.words('english'))

        # Clean text
        df['cleaned_text'] = df['Text'].apply(lambda t: clean_text(t, stop_words))

        print(df[['Text', 'cleaned_text']].head())
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
