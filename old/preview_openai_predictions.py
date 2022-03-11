import json

#open data/wsj_predictions/all.json
filename = 'data/wsj_predictions/vaccine_prompt_filtered_with_price.json'

with open(filename) as f:

    data = json.load(f)

    for article in data:
        print()
        print("********************************************************")
        print()
        print("News:")
        print(article['title'])
        print(article['summary'])

        print()
        print('Prediction: ' + article['Prediction'])

        print()
        print("True Moves")
        for info in article['price_info'].values():
            print(f"{info['ticker']} predicted to go {info['move']}")
            print('prices:', info['prices'] if 'prices' in info else None)
