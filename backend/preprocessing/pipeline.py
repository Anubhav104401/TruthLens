import re
import string
from collections import Counter

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
except Exception:  # pragma: no cover - keeps the API alive if NLTK is unavailable.
    nltk = None
    stopwords = None
    WordNetLemmatizer = None


def _ensure_nltk_resource(resource_name):
    if nltk is None:
        return False

    try:
        nltk.data.find(resource_name)
        return True
    except LookupError:
        package_name = resource_name.split("/")[-1]
        try:
            nltk.download(package_name, quiet=True)
            nltk.data.find(resource_name)
            return True
        except Exception:
            return False

class TextCleaner:
    def __init__(self):
        self.stop_words = set()
        self.lemmatizer = None

        if stopwords and _ensure_nltk_resource("corpora/stopwords"):
            self.stop_words = set(stopwords.words("english"))

        if WordNetLemmatizer and _ensure_nltk_resource("corpora/wordnet"):
            self.lemmatizer = WordNetLemmatizer()

    def strip_source_leakage(self, text):
        # ISOT-style true articles often contain publisher datelines such as
        # "WASHINGTON (Reuters) -". Removing them reduces source leakage.
        text = re.sub(
            r"^\s*[A-Z][A-Z\s.,'-]{2,80}\s+\((?:Reuters|AP)\)\s*[-:]\s*",
            " ",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(r"\b(?:reuters|associated press|ap news)\b", " ", text, flags=re.IGNORECASE)
        return text
    
    def clean(self, text):
        if not isinstance(text, str):
            return ""

        text = self.strip_source_leakage(text)
        
        # 1. Lowercase
        text = text.lower()
        
        # 2. Remove HTML tags
        text = re.sub(r'<[^>]*>', '', text)
        
        # 3. Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # 4. Remove Emails
        text = re.sub(r'\S+@\S+', '', text)
        
        # 5. Remove Numbers (optional, but requested)
        text = re.sub(r'\d+', '', text)
        
        # 6. Remove Punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # 7. Tokenization & 8. Stopword Removal & 9. Lemmatization
        tokens = re.findall(r"[a-z]+", text)
        cleaned_tokens = []
        for word in tokens:
            if word in self.stop_words or len(word) <= 1:
                continue
            if self.lemmatizer:
                word = self.lemmatizer.lemmatize(word)
            cleaned_tokens.append(word)
        
        # 10. Join Tokens
        return " ".join(cleaned_tokens)

class FeatureEngineer:
    FEATURE_COLUMNS = [
        "word_count",
        "sentence_count",
        "avg_sentence_length",
        "avg_word_length",
        "capital_ratio",
        "exclamation_count",
        "question_count",
        "url_count",
        "quote_count",
        "repeated_word_ratio",
        "first_person_ratio",
        "sensational_term_count",
    ]

    SENSATIONAL_TERMS = {
        "shocking",
        "breaking",
        "unbelievable",
        "miracle",
        "exposed",
        "secret",
        "viral",
        "scandal",
        "leaked",
        "conspiracy",
        "hoax",
        "urgent",
    }

    def __init__(self):
        pass
    
    def extract_features(self, text):
        if not isinstance(text, str) or len(text.strip()) == 0:
            return {column: 0 for column in self.FEATURE_COLUMNS}

        # Statistical Features
        words = re.findall(r"\b[\w'-]+\b", text)
        word_count = len(words)
        
        sentences = re.split(r"[.!?]+", text)
        sentences = [s for s in sentences if len(s.strip()) > 0]
        sentence_count = len(sentences) if len(sentences) > 0 else 1
        avg_sentence_length = word_count / sentence_count if sentence_count else 0
        avg_word_length = sum(len(word) for word in words) / word_count if word_count else 0
        
        capital_letters = sum(1 for c in text if c.isupper())
        capital_ratio = capital_letters / len(text) if len(text) > 0 else 0
        
        exclamation_count = text.count("!")
        question_count = text.count("?")
        url_count = len(re.findall(r"https?://\S+|www\.\S+", text))
        quote_count = text.count('"') + text.count("'")

        lower_words = [word.lower() for word in words]
        word_frequencies = Counter(lower_words)
        repeated_words = sum(count - 1 for count in word_frequencies.values() if count > 1)
        repeated_word_ratio = repeated_words / word_count if word_count else 0

        first_person_words = {"i", "me", "my", "mine", "we", "our", "ours", "us"}
        first_person_count = sum(1 for word in lower_words if word in first_person_words)
        first_person_ratio = first_person_count / word_count if word_count else 0

        sensational_term_count = sum(1 for word in lower_words if word in self.SENSATIONAL_TERMS)
        
        return {
            "word_count": word_count,
            "sentence_count": sentence_count,
            "avg_sentence_length": avg_sentence_length,
            "avg_word_length": avg_word_length,
            "capital_ratio": capital_ratio,
            "exclamation_count": exclamation_count,
            "question_count": question_count,
            "url_count": url_count,
            "quote_count": quote_count,
            "repeated_word_ratio": repeated_word_ratio,
            "first_person_ratio": first_person_ratio,
            "sensational_term_count": sensational_term_count,
        }

    def extract_features_df(self, texts):
        import pandas as pd

        features = [self.extract_features(t) for t in texts]
        return pd.DataFrame(features, columns=self.FEATURE_COLUMNS).fillna(0)
