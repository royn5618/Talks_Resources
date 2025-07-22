import pandas as pd
import nltk
import logging
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')


class Config:
    """
    Configuration class for file paths, column names, and NLP resources.
    """
    CSV_PATH = 'data.csv'
    TEXT_COLUMN = 'Text'
    CLEANED_COLUMN = 'cleaned_text'
    STOPWORDS = set(stopwords.words('english'))
    PUNCTUATION = set(string.punctuation)
    LOG_LEVEL = logging.INFO
    LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'


# Configure logger
logging.basicConfig(level=Config.LOG_LEVEL, format=Config.LOG_FORMAT)
logger = logging.getLogger(__name__)


def clean_text(text, stop_words, punctuation):
    """
    Cleans input text by:
    - Lowercasing
    - Tokenizing
    - Removing stopwords and punctuation

    Args:
        text (str): Input text string.
        stop_words (set): Set of stopwords.
        punctuation (set): Set of punctuation characters.

    Returns:
        str: Cleaned and space-joined string.
    """
    tokens = word_tokenize(text.lower())

    cleaned = [word for word in tokens if word not in stop_words and word not in punctuation]
    return ' '.join(cleaned)


def main():
    try:
        logger.info("Reading data from CSV...")
        df = pd.read_csv(Config.CSV_PATH)

        if Config.TEXT_COLUMN not in df.columns:
            raise KeyError(f"Column '{Config.TEXT_COLUMN}' not found in the CSV file.")

        logger.info("Cleaning text data...")
        df[Config.CLEANED_COLUMN] = df[Config.TEXT_COLUMN].apply(
            lambda t: clean_text(t, Config.STOPWORDS, Config.PUNCTUATION)
        )

        logger.info("Cleaning completed. Showing preview:")
        logger.debug(df[[Config.TEXT_COLUMN, Config.CLEANED_COLUMN]].head().to_string(index=False))

    except Exception as e:
        logger.exception(f"An error occurred during processing: {e}")


if __name__ == "__main__":
    main()
