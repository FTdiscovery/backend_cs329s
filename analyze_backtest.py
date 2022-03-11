#open tmp.json
import json

with open('data/wsj_predictions/flight_prompt_with_price.json', 'r') as f:
    data = json.load(f)

print(len(data))

def sign(x):
    if x > 0:
        return 'up'
    elif x == 0:
        return 'flat'
    else:
        return 'down'
counts = {}

# assume we put in 50 stocks or up to $2000 worth for each stock and zero transaction fee
total_earnings = 0

categories = {'Business', 'Economy', 'Heard on the Street', 'Markets', 'Tech', 'Stocks', 'WSJ News Exclusive', 'CFO Journal', 'Pro Bankruptcy', 'CIO Journal'}
dictionary = dict()
earnings_dict = dict()
for c in categories:
    dictionary[c] = [0,0,0,0]
for c in categories:
    earnings_dict[c] = 0

trades = []
balance_history = []

for prediction_obj in data:
    # for category in prediction_obj['categories'][0]:
    #     if category not in categories:
    #         categories.add(category)
    if 'Business' in prediction_obj['categories'][0] or 'Markets' in prediction_obj['categories'][0] or 'Heard on the Street' in prediction_obj['categories'][0] or 'WSJ News Exclusive' in prediction_obj['categories'][0]:
        for d in prediction_obj['price_info'].values():
            if 'move' in d and 'prices' in d and d['move'] and d['prices'] and len(d['prices']) > 1:
                price_move_str = d['move'] + '_' + sign(d['prices'][1] - d['prices'][0])

                earnings = 0
                if d['move'] == "down":
                    # we chose to sell $1000 worth of stock at first time point,
                    # then buy same amount of stock at second time point
                    earnings += (d['prices'][0]-d['prices'][1]) * 10000 / d['prices'][0]

                    for category in prediction_obj['categories'][0]:
                        if category in categories:
                            earnings_dict[category] += (d['prices'][0] - d['prices'][1]) * 20000 / d['prices'][0]
                            if (d['prices'][0]-d['prices'][1])/d['prices'][0] > 0.005:
                                dictionary[category][1] += 1
                            elif (d['prices'][0]-d['prices'][1])/d['prices'][0] < -0.005:
                                dictionary[category][3] += 1
                if d['move'] == "up":
                    earnings += (d['prices'][1] - d['prices'][0]) * 10000 / d['prices'][0]

                    for category in prediction_obj['categories'][0]:
                        if category in categories:
                            earnings_dict[category] += (d['prices'][1] - d['prices'][0]) * 20000 / d['prices'][0]
                            if (d['prices'][1] - d['prices'][0])/d['prices'][0] > 0.005:
                                dictionary[category][0] += 1
                            elif (d['prices'][1] - d['prices'][0])/d['prices'][0] < -0.005:
                                dictionary[category][2] += 1

                total_earnings += earnings
                counts[price_move_str] = counts.get(price_move_str, 0) + 1
                trades.append( (earnings, d['ticker'], prediction_obj['title'] ))
                balance_history.append(total_earnings)

print(counts)
print(balance_history[-1])
for val, ticker, title in sorted(trades, key=lambda x: x[0], reverse=True):
    print(f"{ticker} earned {val} on {title[:50]}")
# print(trades)

for d in dictionary:
    stock = dictionary[d]
    print(d)
    print(stock)
    try:
        print("Accuracy:", (stock[0]+stock[1])/(stock[0]+stock[1]+stock[2]+stock[3]))
    except:
        continue
    print("Earned:", earnings_dict[d])


# plot all of the earnings in trades
import matplotlib.pyplot as plt
import numpy as np

#Plot the balance history vs the index
plt.plot(np.arange(len(balance_history)), balance_history)
plt.xlabel('Time')
plt.ylabel('Balance')
plt.title('Balance History')
plt.show()