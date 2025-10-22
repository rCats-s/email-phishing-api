import string
import numpy as np
import nltk
import re
from urllib.parse import urlparse
from spellchecker import SpellChecker


nltk.download('stopwords')
from nltk.corpus import stopwords
spell = SpellChecker()
stop_words = set(stopwords.words('english'))
urgent_words = [
    "urgent", "immediately", "verify", "update", "account", "password", 
    "click", "login", "important", "attention", "alert", "action", "suspend",
    "reset", "confirm", "security", "limited", "asap", "request", "now", "today",
    "emergency", "critical", "warning", "deadline", "final", "notice", "urgently",
    "hacked", "compromised", "risk", "threat", "safe", "safety", "protect", "protection"
]

def extract_features(some_txt):
    list_txt = some_txt.split()
    print(list_txt)
    id1 = len(list_txt)  #number of words
    id2 = 0  #number of unique words
    id3 = 0  #number of stopwords
    id4 = 0  #number of hyperlinks
    id5 = 0  #number of domains
    id6 = 0  #number of email addresses
    id7 = 0  #number of misspelled words
    id8 = 0  #number of urgent words
    word_counts = {}
    domain_list = []
    potential_misspelled_words = []
    domain_counts = {}
    for word in list_txt:
        clean_word = word.lower().strip(string.punctuation)
        word_counts[clean_word] = word_counts.get(clean_word, 0) + 1
        if re.search("^--", word) and re.search("--$", word): #hyperlink
            id4 += 1
        elif clean_word in stop_words: #stopword
            id3 += 1
        elif clean_word in urgent_words: #urgent word
            id8 += 1
        elif clean_word.startswith("http://") or clean_word.startswith("https://") or clean_word.startswith("ftp://"):
            parsed = urlparse(clean_word)
            domain = parsed.netloc
            if len(domain) > 0:
                domain_list.append(domain)
        elif re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', clean_word):
            id6 += 1
        else:
            potential_misspelled_words.append(clean_word)
    if len(domain_list) > 0:
        for d in domain_list:
            domain_counts[d] = domain_counts.get(d, 0) + 1
    id2 = sum(1 for count in word_counts.values() if count >= 1)
    id5 = sum(1 for count in domain_counts.values() if count >= 1)
    misspelled = spell.unknown(potential_misspelled_words)
    id7 = len(misspelled)

    extr_arr = np.array([id1, id2, id3, id4, id5, id6, id7, id8])
    print("Extracted features:", extr_arr)
    return extr_arr
