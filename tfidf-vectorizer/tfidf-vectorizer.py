import numpy as np
from collections import Counter
import math

def tfidf_vectorizer(documents):
    """
    Build TF-IDF matrix from a list of text documents.
    Returns tuple of (tfidf_matrix, vocabulary).
    """

    tokenized_docs=[doc.split() for doc in documents]
    
    # vocabulary

    all_words=[]
    for doc in documents:
        words=doc.split()
        all_words.extend(words)
        # The key difference is that append always adds one item (even if that item is a list), 
        # while extend unpacks the iterable and adds each element separately.
    

    unique_words=sorted(set(all_words))
    word_index = {word: i for i, word in enumerate(unique_words)}

    # tf idf matrix
    tf_idf_matrix=np.zeros((len(documents), len(unique_words)))

    for doc_i, doc in enumerate(tokenized_docs):
        #print(doc_i,doc)
        word_counts=Counter(doc) 
        for word,count in word_counts.items():
            tf=count/len(doc)
            df_t=0
            for i in tokenized_docs:
                if word in i:
                    df_t+=1
            idf=math.log(len(tokenized_docs)/df_t)
            tf_idf_matrix[doc_i][word_index[word]]=tf*idf
    return tf_idf_matrix,unique_words


 