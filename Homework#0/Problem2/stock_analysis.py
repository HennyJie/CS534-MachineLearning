import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt

data = pd.read_csv("Homework#0/Problem2/AAPL.csv")
close_price = data['Close'].to_numpy()
dates = data['Date'].values

x = [dt.datetime.strptime(d,'%Y-%m-%d').date() for d in dates]

y = close_price
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

locator = mdates.YearLocator()
plt.gca().xaxis.set_major_locator(locator)

plt.plot(x,y)
plt.title('Daily Closing Price of AAPL')
plt.ylabel('daily closing prices')
plt.xlabel('date')
plt.gcf().autofmt_xdate()
plt.savefig("Homework#0/Problem2/daily_closing_price.png")

total_days_count = close_price.shape[0]
daily_return = np.ones(total_days_count-1)
for i in range(total_days_count - 1):
    daily_return[i] = close_price[i+1] / close_price[i]

mean_daily_return = np.mean(daily_return)
variance_daily_return = np.var(daily_return)
print("The mean of the Apple stock’s daily return is: ", mean_daily_return)
print("The variance of the Apple stock’s daily return is: ", variance_daily_return)
print("-----------------------------------------------------------------")

daily_return_list = daily_return.tolist()
daily_return_list.insert(0, None)
data['Return'] = daily_return_list

year_group_dict = {}
for date, ratio in zip(x, daily_return_list):
    if ratio is None:
        continue
    if date.year not in year_group_dict:
        year_group_dict[date.year] = [ratio]
    else:
        year_group_dict[date.year] += [ratio]

for key in year_group_dict.keys():
    # print("Daily returns for the Year {} is: {}".format(key, year_group_dict[key]))
    print("Mean Daily returns for the Year {} is: {}".format(key, np.mean(year_group_dict[key])))
    print("Var Daily returns for the Year {} is: {}".format(key, np.var(year_group_dict[key])))
    print("-----------------------------------------------------------------")

# print("year_group_dict: ", year_group_dict)

# return_data = [dates[1:], daily_return]
# return_data = [[row[i] for row in return_data] for i in range(len(return_data[0]))]
# df = pd.DataFrame(return_data, columns=['Date','Return'])
# group_by_year = df['Return'].groupby(pd.to_datetime(df['Date']).dt.year)
# print(group_by_year.mean())
# print(group_by_year.var())

# plt.show()

