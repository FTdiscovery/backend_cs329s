from alpaca_gpt3_pipeline import create_alpaca_predictions
from wsj_gpt3_pipeline import create_wsj_predictions
from datetime import date, timedelta, datetime
import json
import numpy as np
import itertools
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def max_sum_sim(doc_embedding, word_embeddings, words, top_n, nr_candidates):
    # Calculate distances and extract keywords
    distances = cosine_similarity(doc_embedding, candidate_embeddings)
    distances_candidates = cosine_similarity(candidate_embeddings,
                                             candidate_embeddings)

    # Get top_n words as candidates based on cosine similarity
    words_idx = list(distances.argsort()[0][-nr_candidates:])
    words_vals = [candidates[index] for index in words_idx]
    distances_candidates = distances_candidates[np.ix_(words_idx, words_idx)]

    # Calculate the combination of words that are the least similar to each other
    min_sim = np.inf
    candidate = None
    for combination in itertools.combinations(range(len(words_idx)), top_n):
        sim = sum([distances_candidates[i][j] for i in combination for j in combination if i != j])
        if sim < min_sim:
            candidate = combination
            min_sim = sim

    return [words_vals[idx] for idx in candidate]


if __name__ == "__main__":

    day = datetime.today()-timedelta(days=5)
    create_alpaca_predictions(day)
    create_wsj_predictions(day)

    date_str = day.strftime('%Y-%m-%d')
    combined = []

    file_name = f"data/alpaca_predictions/{date_str}_with_price.json"
    with open(file_name) as f:
        combined += json.load(f)

    file_name = f"data/wsj_predictions/{date_str}_with_price.json"
    with open(file_name) as f:
        combined += json.load(f)

    doc = [d['summary'] for d in combined]
    n_gram_range = (1, 3)
    stop_words = "english"
    top_n = 3
    model_distilbert = SentenceTransformer('distilbert-base-nli-mean-tokens')
    model_xlmr = SentenceTransformer('sentence-transformers/xlm-r-distilroberta-base-paraphrase-v1')

    for i, d in enumerate(doc):
        count_stop = CountVectorizer(ngram_range=n_gram_range, stop_words=stop_words).fit([d])
        candidates = count_stop.get_feature_names()
        print(i)
        if len(candidates) > 5:
            doc_embedding_d = model_distilbert.encode([d])
            candidate_embeddings_d = model_distilbert.encode(candidates)
            doc_embedding_x = model_xlmr.encode([d])
            candidate_embeddings_x = model_xlmr.encode(candidates)

            doc_embedding = doc_embedding_d + doc_embedding_x
            candidate_embeddings = candidate_embeddings_d + candidate_embeddings_x

            keywords = max_sum_sim(doc_embedding, candidate_embeddings, candidates, top_n=top_n, nr_candidates=20)
            combined[i]['keywords'] = keywords

    """
    SAVE FILE
    """
    file_name = f"data/combined_predictions/{date_str}_sentiments_keywords.json"
    with open(file_name, 'w') as f:
        json.dump(combined, f)
    print("A total of", len(combined), "articles were predicted on for", date_str, ".")