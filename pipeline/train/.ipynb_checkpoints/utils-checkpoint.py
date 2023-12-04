import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter, defaultdict
from tqdm import tqdm

def preprocess_text(text):
    words = word_tokenize(text)
    words = [word.lower() for word in words]
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word.isalnum() and word not in stop_words]
    return words

def build_text_count_and_dict(words):
    word_count = Counter(words)
    word2ind = defaultdict(lambda: len(word_count))
    for i, word in enumerate(word_count.keys()):
        word2ind[word] = i
    return word_count, word2ind

def BOW(text, word2ind):
    words = preprocess_text(text)
    bow = [0] * (len(word2ind.keys()) + 1)
    for word in words:
        bow[word2ind[word]] += 1
    return bow

def compute_tfidf(corpus):
    vectorizer = TfidfVectorizer(tokenizer=preprocess_text, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(corpus)
    return tfidf_matrix, vectorizer.vocabulary_

if __name__ == '__main__':
    train_file = 'dataset/processed/train_data.csv'
    data = pd.read_csv(train_file)
    steps_text = data.steps.tolist()

    print("Preprocessing text and building vocabulary...")
    words = preprocess_text(' '.join(steps_text))
    word_count, word2ind = build_text_count_and_dict(words)

    bow_train = []
    for text in tqdm(steps_text, desc="Building BOW representations"):
        bow_train.append(BOW(text, word2ind))

    print("Computing TF-IDF...")
    tfidf_matrix, tfidf_vocab = compute_tfidf(steps_text)

    print("BOW representations and TF-IDF matrix computed.")
