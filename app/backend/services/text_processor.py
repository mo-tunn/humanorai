import re
import nltk
from nltk.corpus import stopwords
import warnings

# NLTK Setup
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords', quiet=True)

def strict_clean(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text) 
    words = text.split()
    stop_words = set(stopwords.words('english'))
    custom_stops = {'abstract', 'summary', 'title', 'introduction', 'conclusion', 'paper', 'keywords'} 
    all_stops = stop_words.union(custom_stops)
    clean_words = [w for w in words if w not in all_stops and len(w) > 2]
    return " ".join(clean_words)
