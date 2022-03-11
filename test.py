from alpaca_gpt3_pipeline import create_alpaca_predictions
from wsj_gpt3_pipeline import create_wsj_predictions
from datetime import date, timedelta, datetime
import json
import numpy as np
#
# day = datetime.today()-timedelta(days=1)
# date_str = day.strftime('%Y-%m-%d')
# combined = []
#
# file_name = f"data/alpaca_predictions/{date_str}_with_price.json"
# with open(file_name) as f:
#     combined += json.load(f)
#
# file_name = f"data/wsj_predictions/{date_str}_with_price.json"
# with open(file_name) as f:
#     combined += json.load(f)
#
# trades = 0
# for elem in combined:
#     trades += len(elem["sentiments"])
# print(trades/len(combined))

combined = []

file_name = f"data/combined_predictions/2022-03-04_sentiments_keywords.json"
with open(file_name) as f:
    combined += json.load(f)

file_name = f"data/combined_predictions/2022-03-07_sentiments_keywords.json"
with open(file_name) as f:
    combined += json.load(f)

file_name = f"data/combined_predictions/2022-03-08_sentiments_keywords.json"
with open(file_name) as f:
    combined += json.load(f)

file_name = f"data/combined_predictions/2022-03-09_sentiments_keywords.json"
with open(file_name) as f:
    combined += json.load(f)

file_name = "data/combined_predictions/2022-03-04to09_all_features.json"
with open(file_name, 'w') as f:
    json.dump(combined, f)

print(len(combined))