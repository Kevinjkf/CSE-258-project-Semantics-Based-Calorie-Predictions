import pandas as pd
from utils import preprocess_text,build_text_count_and_dict,BOW

import nltk

nltk.download('stopwords')
nltk.download('punkt')

train_file = 'dataset/processed/train_data.csv'
data = pd.read_csv(train_file)
steps_text = data.steps.tolist()
words = preprocess_text(''.join(steps_text))
word_count,word2ind = build_text_count_and_dict(words)

bow_train = []
for text in steps_text:
    bow_train.append(BOW(text,word2ind))

print()