#pgm 5
!pip install gensim nltk

import gensim.downloader as api

# Load GloVe model
model = api.load("glove-wiki-gigaword-50")

# Define a function to construct a paragraph using similar words
def construct_paragraph(seed_word, similar_words):
    paragraph = f""" 
    In the spirit of {seed_word}, one might embark on an unforgettable 
    {similar_words[0][0]} to distant lands. Every {similar_words[1][0]} 
    brings new challenges and opportunities for {similar_words[2][0]}.
    Through perseverance and courage, the {similar_words[3][0]} becomes 
    a tale of triumph, much like an {similar_words[4][0]}.
    """
    return paragraph

# Input word
input_word = "adventure"

# Get top 5 similar words
similar_words = model.most_similar(input_word, topn=5)

# Construct and print the paragraph
paragraph = construct_paragraph(input_word, similar_words)
print(paragraph)
