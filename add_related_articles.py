import copy
import numpy as np
import os
import faiss
import json

os.environ['KMP_DUPLICATE_LIB_OK']='True'

from transformers import DistilBertTokenizer, DistilBertModel
from tqdm import tqdm

VEC_DIM = 768
HARD_STOP = float('inf')
LOADED = False
K = 4

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained("distilbert-base-uncased")

ARTICLES_FNAME = "data/combined_predictions/2022-03-09_sentiments_keywords.json"
print(ARTICLES_FNAME)

def get_vector(text):
    encoded_input = tokenizer(text, return_tensors='pt')
    vec = model(**encoded_input)
    vec = vec.last_hidden_state.detach().numpy()
    vec = vec.reshape(vec.shape[1], vec.shape[2])
    vec = np.mean(vec, axis=0)
    return vec

all_articles = []

with open(ARTICLES_FNAME) as f:
    all_articles = json.load(f)

all_articles = all_articles[::-1] # Most to least recent

for i, article in enumerate(all_articles):
    article['id'] = i

if not LOADED:
    print("Loading vectors...")
    vecs = []
    for i, article in tqdm(enumerate(all_articles)):
        vec = get_vector(article['title'])
        vecs.append(vec)
        if i == HARD_STOP: break

    vecs = np.array(vecs)

    np.save('data.npy', vecs)

vecs = np.load('data.npy')

index = faiss.IndexFlatL2(VEC_DIM)

index.add(vecs)

def find_related(id):
    id = int(id)
    article = all_articles[id]
    neighbors = []

    vec = get_vector(article['title'])
    vec = vec.reshape(1, -1)
    d, inds = index.search(vec, K)
    inds = inds[0]
    for ix in inds:
        nbr = all_articles[ix]
        if nbr['link'] == article['link']: continue # Avoid duplication
        neighbors.append(copy.deepcopy(nbr))
    return neighbors

for i, article in enumerate(all_articles):
    neighbors = find_related(i)
    article['nbrs'] = neighbors

if __name__ == '__main__':
    for i, article in enumerate(all_articles):
        print()
        print()
        print()
        print(f"Searching article {article['title']}")
        neighbors = find_related(i)
        for j, nbr in enumerate(neighbors):
            print(f"Neighbor {j + 1}")
            print(f"Article title: {nbr['title']}")
            print()

    file_name = "data/combined_predictions/2022-03-09_all_features.json"
    with open(file_name, 'w') as f:
        json.dump(all_articles, f)
