import numpy as np
import pandas as pd
import requests
import math
#import tensorflow
import datetime as dt
from datetime import date,timedelta
from pandas_datareader import data as web
import yfinance as yf
from scipy import stats

def save_to_csv_from_yahoo(ticker, syear, smonth, sday, eyear, emonth, eday):
    start = dt.datetime(syear, smonth, sday)
    end = dt.datetime(eyear, emonth, eday)
    df = web.DataReader(ticker, 'yahoo', start, end)
    df.to_csv('data.csv')
    return df

def get_valid_dates(df, sdate, edate):
    mask = (df.index > sdate) & (df.index <= edate)
    sm_df = df.loc[mask]
    sm_date = sm_df.index.min()
    last_date = sm_df.index.max()
    #print(sm_date, " ", last_date)

    return sm_date, last_date

def roi_between_dates(df, sdate, edate):
    try:
        s,e=get_valid_dates(df,sdate,edate)
        start_val = df.loc[s, 'Adj Close']
        end_val = df.loc[e, 'Adj Close']
        roi = ((end_val - start_val) / start_val)
    except Exception:
        print("Data Corrupted")
    else:
        return roi

#curr price
def curr_price(ticker):
    today = date.today()
    d1 = today.strftime("%Y-%m-%d")
    end_date = d1
    d2 = date.today() - timedelta(days=1)
    d2 = d2.strftime("%Y-%m-%d")
    start_date = d2

    data = yf.download(ticker,
                          start=start_date,
                          end=end_date,
                          progress=False)
    return data['Adj Close'].values[0]

def yearly_return(ticker):
    today = date.today()
    d1 = today.strftime("%Y-%m-%d")
    end_date = d1
    d2 = date.today() - timedelta(days=365)
    d2 = d2.strftime("%Y-%m-%d")
    start_date = d2

    data = yf.download(ticker,
                       start=start_date,
                       end=end_date,
                       progress=False)
    df = pd.DataFrame(data)
    return roi_between_dates(df,start_date,end_date)

def halfyearly_return(ticker):
    today = date.today()
    d1 = today.strftime("%Y-%m-%d")
    end_date = d1
    d2 = date.today() - timedelta(days=183)
    d2 = d2.strftime("%Y-%m-%d")
    start_date = d2

    data = yf.download(ticker,
                       start=start_date,
                       end=end_date,
                       progress=False)
    df = pd.DataFrame(data)
    return roi_between_dates(df,start_date,end_date)

def threemonth_return(ticker):
    today = date.today()
    d1 = today.strftime("%Y-%m-%d")
    end_date = d1
    d2 = date.today() - timedelta(days=93)
    d2 = d2.strftime("%Y-%m-%d")
    start_date = d2

    data = yf.download(ticker,
                       start=start_date,
                       end=end_date,
                       progress=False)
    df = pd.DataFrame(data)
    return roi_between_dates(df,start_date,end_date)

def onemonth_return(ticker):
    today = date.today()
    d1 = today.strftime("%Y-%m-%d")
    end_date = d1
    d2 = date.today() - timedelta(days=31)
    d2 = d2.strftime("%Y-%m-%d")
    start_date = d2

    data = yf.download(ticker,
                       start=start_date,
                       end=end_date,
                       progress=False)
    df = pd.DataFrame(data)
    return roi_between_dates(df,start_date,end_date)


stocks = pd.read_csv('sp_500_stocks.csv')
my_columns = ['Ticker','Number Of Shares to Buy','Market Cap']
final_dataframe = pd.DataFrame(columns = my_columns)
final_dataframe['Ticker']=stocks['Ticker']
final_dataframe.set_index('Ticker',inplace=True)
final_dataframe.loc['AAPL','Number Of Shares to Buy']=1

market_data=[]
for ticker in stocks['Ticker']:
    try:
        market_data.append(int(web.get_quote_yahoo(ticker)['marketCap']))
    except:
        market_data.append(np.nan)
print(market_data)
final_dataframe['Market Cap']=market_data
final_dataframe
final_dataframe.to_csv('framework.csv')

df=pd.read_csv('framework.csv')
df=df[['Ticker','Number Of Shares to Buy','Market Cap']]
df.set_index('Ticker',inplace=True)
df = df[df['Market Cap'].notna()]
df.to_csv('framework.csv')

df=pd.read_csv('framework.csv')
try:
    portfolio_size=float(input('size '))
except ValueError:
    print('invalid size given')

#equal weight snp 500 fund index
position=portfolio_size/len(df)

#number_of_shares=position//curr_price('AAPL')
#print(number_of_shares)
print(position)
for i in range(0,len(df)):
    try:
        curr=curr_price(df.iloc[i,0])
        number_of_shares=position//curr
        print(number_of_shares)
        df.iloc[i,1]=number_of_shares
    except:
        print(df.iloc[i,0])

df.to_csv('equal_weight_snp500.csv')

df=pd.read_csv('equal_weight_snp500.csv')
df=df[['Ticker','Number Of Shares to Buy','Market Cap']]
df.set_index('Ticker',inplace=True)
df = df[df['Number Of Shares to Buy'].notna()]
print(df)
df.to_csv('equal_weight_snp500.csv')

'''our universe is the left over stocks'''

'''building a quantitative momentum strategy
"Momentum investing" means investing in the stocks that have increased in price the most.
For this project, we're going to build an investing strategy that selects the 50 stocks with the highest price momentum. 
From there, we will calculate recommended trades for an equal-weight portfolio of these 50 stocks.
'''

df=pd.read_csv('equal_weight_snp500.csv')
stocks=df['Ticker'].values

ret=[0]*len(stocks)
df['returns']=ret

today = date.today()
d1 = today.strftime("%Y-%m-%d")
end_date = d1
d2 = date.today() - timedelta(days=365)
d2 = d2.strftime("%Y-%m-%d")
start_date = d2

print(df)
for i in range(len(stocks)):
    ticker=str(df.iloc[i, 0])

    data = yf.download(ticker,
                       start=start_date,
                       end=end_date,
                       progress=False)
    stockdf = pd.DataFrame(data)
    df.iloc[i,3]=roi_between_dates(stockdf, start_date, end_date)

df.to_csv('momentum.csv')
df=pd.read_csv('momentum.csv')
final=df[['Ticker','Number Of Shares to Buy','Market Cap','returns']]
#removing low momentum stocks

final=df.sort_values('returns',ascending=False)[:50]
final=final[['Ticker','Number Of Shares to Buy','Market Cap','returns']]
#final=final.set_index('Ticker')



try:
    portfolio_size=float(input('size '))
except ValueError:
    print('invalid size given')

#equal weight snp 500 fund index
position=portfolio_size/len(final)

#number_of_shares=position//curr_price('AAPL')
#print(number_of_shares)
print(position)
for i in range(0,len(final)):
    try:
        curr=curr_price(final.iloc[i,0])
        number_of_shares=position//curr
        print(i)
        final.iloc[i,1]=number_of_shares
    except:
        print(final.iloc[i,0])
print(final)

final.to_csv('not_good_momentum.csv',index=False)

'''Real-world quantitative investment firms differentiate between "high quality" and "low quality" momentum stocks:

    High-quality momentum stocks show "slow and steady" outperformance over long periods of time
    Low-quality momentum stocks might not show any momentum for a long time, and then surge upwards.

The reason why high-quality momentum stocks are preferred is because low-quality momentum can often be cause by short-term news that is unlikely to be repeated in the future (such as an FDA approval for a biotechnology company).

To identify high-quality momentum, we're going to build a strategy that selects stocks from the highest percentiles of:

    1-month price returns
    3-month price returns
    6-month price returns
    1-year price returns

Let's start by building our DataFrame. You'll notice that I use the abbreviation hqm often. It stands for high-quality momentum.'''

df=pd.read_csv('equal_weight_snp500.csv')
stocks=df['Ticker'].values

hqm_columns = [
                'Ticker',
                'Price',
                'Number of Shares to Buy',
                'One-Year Price Return',
                'One-Year Return Percentile',
                'Six-Month Price Return',
                'Six-Month Return Percentile',
                'Three-Month Price Return',
                'Three-Month Return Percentile',
                'One-Month Price Return',
                'One-Month Return Percentile',
                'HQM Score'
                ]

hqm_dataframe = pd.DataFrame(columns = hqm_columns)
hqm_dataframe['Ticker']=stocks
price=[0]*len(hqm_dataframe)

for i in range(len(price)):
    price[i]=curr_price(stocks[i])
hqm_dataframe['Price']=price
print(hqm_dataframe)
hqm_dataframe.to_csv('hq_momentum.csv',index=False)

df=pd.read_csv('equal_weight_snp500.csv')
stocks=df['Ticker'].values
df=pd.read_csv('hq_momentum.csv')
yearly=[0]*len(df)
half=[0]*len(df)
three=[0]*len(df)
one=[0]*len(df)

for i in range(len(yearly)):
    yearly[i]=yearly_return(stocks[i])
    half[i]=halfyearly_return(stocks[i])
    three[i]=threemonth_return(stocks[i])
    one[i]=onemonth_return(stocks[i])


df['One-Year Price Return']=yearly
df['Six-Month Price Return']=half
df['Three-Month Price Return']=three
df['One-Month Price Return']=one


print(df)
df.to_csv('hq_returns_momentum.csv',index=False)

df=pd.read_csv('hq_returns_momentum.csv')
stocks=df['Ticker'].values

time_periods = [
                'One-Year',
                'Six-Month',
                'One-Month',
                ]

for row in df.index:
    for time_period in time_periods:
        df.loc[row, f'{time_period} Return Percentile'] = \
            stats.percentileofscore(df[f'{time_period} Price Return'], df.loc[row, f'{time_period} Price Return'])/100

df=df.drop(['Three-Month Return Percentile','Three-Month Price Return'], axis=1)

#Print the entire DataFrame
print(df.tail().to_string())
df.to_csv('percentiles.csv')

df=pd.read_csv('percentiles.csv')
#we make our hqm score to be the mean of all the other percentile scores


from statistics import mean
time_periods = ['One-Year',
                'Six-Month',
                'One-Month']
for row in df.index:
    momentum_percentiles = []
    for time_period in time_periods:
        momentum_percentiles.append(df.loc[row, f'{time_period} Return Percentile'])
    df.loc[row, 'HQM Score'] = mean(momentum_percentiles)


df=df.sort_values(by = 'HQM Score', ascending = False)
df = df[:51]
df=df.drop(['Unnamed: 0'],axis=1)
print(df.to_string())
try:
    portfolio_size=float(input('size '))
except ValueError:
    print('invalid size given')

#equal weight snp 500 fund index
position=portfolio_size/len(df)

#number_of_shares=position//curr_price('AAPL')
#print(number_of_shares)
print(position)
for i in range(0,len(df)):
    try:
        curr=curr_price(df.iloc[i,0])
        number_of_shares=position//curr
        df.iloc[i,2]=number_of_shares
    except:
        print(df.iloc[i,0])
print(df.to_string())
df.to_csv('final_momentum_strat.csv')





