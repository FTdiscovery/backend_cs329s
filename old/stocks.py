import pandas as pd
import json
df = pd.read_csv('data/tickers/NYSE.csv')
df2 = pd.read_csv('data/tickers/NASDAQ.csv')

tickers = dict(zip(df.Symbol, df.Name)) | dict(zip(df2.Symbol, df2.Name))
print(list(tickers.keys()))

with open('data/tickers/ticker-dict.json', "w") as f:
    json.dump(tickers, f)


