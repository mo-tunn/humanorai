import re
import nltk
from nltk.corpus import stopwords
import warnings

# NLTK Setup
try:
    # We check for the resource
    nltk.data.find('corpora/stopwords')
except (LookupError, Exception):
    # If not found, or any other error occurs, we download it
    print("Stopwords not found. Downloading...")
    nltk.download('stopwords', quiet=True)

def strict_clean(text):
    # 1. Lowercase and string conversion
    text = str(text).lower()
    
    # 2. Remove everything except lowercase letters and spaces
    text = re.sub(r'[^a-z\s]', '', text) 
    
    # 3. Tokenize by whitespace
    words = text.split()
    
    # 4. Define Stopwords
    stop_words = set(stopwords.words('english'))
    custom_stops = {'abstract', 'summary', 'title', 'introduction', 'conclusion', 'paper', 'keywords'} 
    all_stops = stop_words.union(custom_stops)
    
    # 5. Filter: Remove stops and words shorter than 3 characters
    clean_words = [w for w in words if w not in all_stops and len(w) > 2]
    
    return " ".join(clean_words)

# Quick Test
example = "This is a Summary of the research paper! It includes an introduction."
print(strict_clean(example))
# Output: "research includes"