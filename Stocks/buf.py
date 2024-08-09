import time
import pandas as pd
from textblob import TextBlob
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

# URL of the news website (e.g., BBC Technology news)
NEWS_URL = 'https://www.bbc.com/news/technology'

# Set up Selenium WebDriver
options = webdriver.ChromeOptions()
options.add_argument('--headless')
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

def get_articles():
    driver.get(NEWS_URL)
    time.sleep(3)  # Wait for page to load
    articles = driver.find_elements(By.CLASS_NAME, 'sc-4fedabc7-3.zTZri')
    
    news_data = []
    for article in articles:
        title = article.text
        link = article.get_attribute('href')
        news_data.append((title, link))
        
    return news_data

def analyze_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

def main():
    news_data = get_articles()
    news_df = pd.DataFrame(news_data, columns=['Title', 'Link'])
    
    news_df['Sentiment'] = news_df['Title'].apply(analyze_sentiment)
    
    # Print news articles with their sentiment score
    for index, row in news_df.iterrows():
        print(f"Title: {row['Title']}")
        print(f"Link: {row['Link']}")
        print(f"Sentiment: {row['Sentiment']}")
        print('-' * 80)
    
    # Save the data to a CSV file
    news_df.to_csv('news_sentiment_analysis.csv', index=False)
    driver.quit()

if __name__ == '__main__':
    main()