# import quandl
import os
import pandas as pd
import pickle
from bs4 import BeautifulSoup
import requests
from start_end_date import start_date, end_date


# get api key from quandl
# quandl.ApiConfig.api_key='E--yyUyma2yixeocWLtG'

# set start and end dates
startdate = start_date
enddate = end_date

# print(startdate, enddate)
def nifty_50_list():
    url = 'https://en.wikipedia.org/wiki/NIFTY_50'
    resp = requests.get(url)
    soup = BeautifulSoup(resp.text, 'html.parser')          #parse the html content of the website

    table = soup.find('table', {'id': 'constituents'},'tbody') #find the table with id 'constituents' found with inspect in website

    tickers = []
    for row in table.findAll('tr')[1:]:     #[1:] to skip the header row
        ticker = row.findAll('td')[0].text  #[0] to get the first column which contains the ticker names
        # print(f"ticker{ticker}")
        tickers.append(ticker[:-1])         #[:-1] to remove the '\n' character at the end of the ticker name
    
    tickers.remove('BAJAJ-AUTO')    #removing tickers which are present with spl characters , which are not accepte in quandl
    tickers.remove('M&M')
    tickers.append('BAJAJ_AUTO')
    tickers.append('MM')
    tickers.sort()
    tickers.append('NIFTY')         #Fetching data for NIFTY50 index whose price we want to predict

    with open("nifty50_list.pickle","wb") as f:     #save the tickers to a pickle file by converting parser object into a byte stream
        pickle.dump(tickers,f)
        
    return tickers


#function to obtain the list of NIFTY50 stocks from the pickle file if it exists, else scrap the website and save the tickers to a pickle file
def get_nifty50_list(scrap=False):
    if scrap:
        tickers=nifty_50_list()
    else:
        with open("nifty50_list.pickle","rb") as f:
            tickers=pickle.load(f)
    return tickers

print("The tickers for the NIFTY50 list", get_nifty50_list(True))


#function to fetch stock prices from Quandl and then storing them to avoid making duplicate calls to Quandl API
def getStockdataFromQuandl(ticker):
    quandl_code="NSE/"+ticker
    try:
        if not os.path.exists(f'stock_data/{ticker}.csv'):
          data=quandl.get(quandl_code,start_date=startdate,end_date=enddate)
          data.to_csv(f'stock_data/{ticker}.csv')
        else:
            print(f"stock data for {ticker} already exists")
    except quandl.errors.quandl_error.NotFoundError as e:
        print(ticker)
        print(str(e))
