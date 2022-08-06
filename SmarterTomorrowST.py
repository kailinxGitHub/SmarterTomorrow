import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import wordcloud
from wordcloud import WordCloud
from wordcloud import ImageColorGenerator
from wordcloud import STOPWORDS
import nltk
nltk.download('vader_lexicon')
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.probability import FreqDist
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import twitter_samples
import tqdm
from tqdm.notebook import tqdm
import transformers
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
import scipy
from scipy.special import softmax

# Miscellaneous
    # CSV converter
@st.cache
def convert_df(file):
    return file.to_csv().encode('utf-8')
    
# Header
header = st.container()
with header: 
    st.title('SmarterTomorrow')
    st.caption('**Description**: ')
    st.caption("SmarterTomorrow is an app that allows the users to filter tweets involving their researching target company and get stats from the data at a clicik of a button!")

#Access Limit (once a day from one IP)



# Search
st.header("Search")
search_form = st.form("search_form")
with search_form: 
    comp_name = st.text_input("Company Name (aka Twitter Tag)", placeholder='e.g. Google, Tesla')
    # duration = st.selectbox("Choose a duration", ('Recent One Quater', 'Recent Two Quaters', 'Recent Year', 'Recent Two Year'))
    selected_apis = st.multiselect('Select APIs', ['Twitter', 'Another Platform'], default='Twitter')
    analyser = st.selectbox("Choose an Analyser", ('VADER: Accurate & Fast', 'RoBERTa: Premium Accuracy & Very Slow'))
    checkbox_val = st.checkbox("I agree to the Terms & Conditions, and that this is not a Financial Advisor!")
    searched = st.form_submit_button("Search")
    if searched:
        if len(comp_name) != 0 and checkbox_val == True:
            # st.write("Company Name: ", comp_name)
            # st.write("Duration: ", duration,)
            # st.write("Analyser: ", analyser)
            # st.write("Checkbox: ", checkbox_val)
            st.write("Successful!")
        elif len(comp_name) == 0 and checkbox_val == True:
            st.warning("**Please Enter a Company Name**")
        elif len(comp_name) != 0 and checkbox_val == False:
            st.warning("**Please agree to the Terms & Conditions!**")
        elif len(comp_name) == 0 and checkbox_val == False:
            st.warning("**No Company Input, and agree to the Terms & Conditions! Try again!**")

### Another Platform
import geocoder
import streamlit as st
location = geocoder.ip("me")
ip_address = location.ip
st.write(ip_address)

### Twitter
def twitter():
    #Twitter API input
    twitter_section = st.header("Twitter")
    with twitter_section:

        import tweepy
        import config

        # client = tweepy.Client(bearer_token= "AAAAAAAAAAAAAAAAAAAAAGG%2BfAEAAAAA8LxwtsjaNZqeMZ3D1oIljvRh7gY%3DK71AEHJOlGtRH5jUv5xYG6cbrhBs8jl8Ft9YiAdl4GBKuBcLwF")

        # name of the account/keyword
        # query = comp_name

        # response = client.search_recent_tweets(query=query, max_results=100, tweet_fields=["created_at", "lang"], expansions=["author_id"])

        # users = {u['id']: u for u in response.includes['users']}

        # full_table = []

        # for tweet in response.data:

        #     language = tweet.lang
        #     if language == "en":
        #         element_table = []

        #     # user
        #         user = users[tweet.author_id]
        #         username = user.username
        #         element_table.append(username)

        #     # tweet
        #         tweet_id = tweet.id
        #         element_table.append(tweet_id)

        #         tweet_text = tweet.text
        #         element_table.append(tweet_text)

        #         full_table.append(element_table)
        #     else:
        #         continue

        # full_pd = np.array(full_table)
        # df = pd.DataFrame(full_pd)
        # st.write(df)
        #     # cleanup
        # # df = df.drop(['Unnamed: 0'], axis=1)

    searched_status = True


    # Analysis
    start_analyser = st.checkbox('Start The Data Section!')
    if searched_status == True and start_analyser == True:
        # Data
        st.header("Data")
        data_preparation = st.container()
        with data_preparation:
            st.write('Total: ', df.shape[0], 'Tweets')
            # view
            if st.checkbox('Show full raw data'):
                st.subheader('Raw data')
                st.write(df)
            st.subheader("Random samples of data")
            st.write(df.sample(n=3))

        dataset = st.container()
        with dataset: 
            text = " ".join(i for i in df["tweet_text"])
            stopwords = set(STOPWORDS)
            wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis("off")
            plt.show()
            st.pyplot()
            st.set_option('deprecation.showPyplotGlobalUse', False)

            # wordcloud

        # Sentiment
        sentiment = st.container()
        with sentiment: 
            st.header("Sentiment Analysis")
            
            ## Analysers
                # Vader
            def vader_analyser():
                st.subheader("VADER: Socia Media Dedicated")
                if st.checkbox("Start Analysing!"): 
                    analyzer = SentimentIntensityAnalyzer()
                    # First back up the values in 'dfV'
                    dfV = pd.DataFrame(df, columns = ['username', 'tweet_id', 'tweet_text', 'vader_neg', 'vader_neu', 'vader_pos', 'vader_compound'])
                    dfV['vader_neg'] = df['tweet_text'].apply(lambda x:analyzer.polarity_scores(x)['neg'])
                    dfV['vader_neu'] = df['tweet_text'].apply(lambda x:analyzer.polarity_scores(x)['neu'])
                    dfV['vader_pos'] = df['tweet_text'].apply(lambda x:analyzer.polarity_scores(x)['pos'])
                    dfV['vader_compound'] = df['tweet_text'].apply(lambda x:analyzer.polarity_scores(x)['compound'])

                    # VADER only
                    st.subheader('VADER Table')
                    dfVO = pd.DataFrame(dfV, columns=['vader_compound', 'vader_neg', 'vader_neu', 'vader_pos', 'tweet_text'])

                    if st.checkbox('Show full data table with VADER values'):
                        st.subheader('Full Table With VADER')
                        st.write(dfV.head())
                        vader_csv = convert_df(dfV)
                        st.download_button(
                            "Press to Download",
                            vader_csv,
                            "vader_full_table.csv",
                            "text/csv",
                            key='download-csv'
                        )
                    else:
                        st.write(dfVO.head())

                    # mean
                    vader_mean = dfVO["vader_compound"].mean()
                    vader_mean_list = vader_mean.tolist()
                    vader_mean_slider = st.slider('-1.00: Negative, 0.00: Neutral, 1.00: Positive', min_value=-1.00, max_value=1.00, value=vader_mean_list)
                    st.write('Total Sentiment: ', vader_mean_slider)

                    # pie chart
                    vader_pie_labels = 'Negative', 'Neutral', 'Positive'
                    vader_neg_lines = (dfVO["vader_neg"] != 0).sum()
                    vader_neu_lines = (dfVO["vader_neu"] != 0).sum()
                    vader_pos_lines = (dfVO["vader_pos"] != 0).sum()
                    vader_lines_sum = vader_neg_lines+vader_neu_lines+vader_pos_lines
                    vader_neg_perc = vader_neg_lines/vader_lines_sum
                    vader_neu_perc = vader_neu_lines/vader_lines_sum
                    vader_pos_perc = vader_pos_lines/vader_lines_sum
                    sizes = [vader_neg_perc, vader_neu_perc, vader_pos_perc]
                    explode = (0, 0, 0)
                    vader_fig, vader_ax = plt.subplots()
                    vader_ax.pie(sizes, explode=explode, labels=vader_pie_labels, autopct='%1.1f%%',
                            shadow=True, startangle=90)
                    vader_ax.axis('equal')
                    st.pyplot(vader_fig)
                    
                    #results
                    vader_mean_str = str(vader_mean)
                    vader_cut_str = vader_mean_str[:6]
                    st.warning("This is not a Financial Advisor")
                    if vader_mean <= -0.5:
                        st.subheader("Your Final Sentiment Is: " + vader_cut_str)
                        st.subheader("It's very likely that the company value will soon been DECREASING at a rapid rate!!! Be Careful!")
                    elif vader_mean > -0.5 and vader_mean < 0:
                        st.subheader("Your Final Sentiment Is: " + vader_cut_str)
                        st.subheader("The near future of the company does not look too bright. Be Cautious")
                    elif vader_mean > 0 and vader_mean < 0.5:
                        st.subheader("Your Final Sentiment Is: " + vader_cut_str)
                        st.subheader("The company will have a steady growth in the near future!")
                    else: 
                        st.subheader("Your Final Sentiment Is: " + vader_cut_str)
                        st.subheader("The company's value is going to SKYROCKET very soon!")
            
                #RoBERTa
            def roberta_analyser():
                st.subheader("RoBERTa: Pretrained NLP")
                if st.checkbox("Start Analysing!"): 
                    MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
                    tokenizer = AutoTokenizer.from_pretrained(MODEL)
                    # from tensorflow import BertTokenizer
                    # tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case = True)
                    model = AutoModelForSequenceClassification.from_pretrained(MODEL)

                    # Process
                    dfR = pd.DataFrame(df, columns = ['username', 'tweet_id', 'tweet_text', 'roberta_neg', 'roberta_neu', 'roberta_pos', 'roberta_compound'])
                    roberta_negL = []
                    roberta_neuL = []
                    roberta_posL = []
                    roberta_compoundL = []
                    for i in df["tweet_text"]:
                        # tokenize
                        encoded_text = tokenizer(i, return_tensors='pt')
                        output = model(**encoded_text)
                        scores = output[0][0].detach().numpy()
                        scores = softmax(scores)
                        roberta_neg = scores[0]
                        roberta_negL.append(roberta_neg)
                        roberta_neu = scores[1]
                        roberta_neuL.append(roberta_neu)
                        roberta_pos = scores[2]
                        roberta_posL.append(roberta_pos)
                        roberta_compound_sum = roberta_neu + roberta_pos - roberta_neg
                        roberta_compound = float(roberta_compound_sum)
                        roberta_compoundL.append(roberta_compound)
                    dfR = dfR.assign(roberta_neg=roberta_negL)
                    dfR = dfR.assign(roberta_neu=roberta_neuL)
                    dfR = dfR.assign(roberta_pos=roberta_posL)
                    dfR = dfR.assign(roberta_compound=roberta_compoundL)

                    #RoBERTa only
                    st.subheader('RoBERTa Table')
                    dfRO = pd.DataFrame(dfR, columns = ['roberta_compound', 'roberta_neg', 'roberta_neu', 'roberta_pos', 'tweet_text'])
                    
                    # First back up the values in 'dfR'
                    if st.checkbox('Show full data table with RoBERTa values'):
                        st.subheader('Full Table With RoBERTa')
                        st.write(dfR.head())
                        vader_csv = convert_df(dfR)
                        st.download_button(
                            "Press to Download",
                            vader_csv,
                            "roberta_full_table.csv",
                            "text/csv",
                            key='download-csv'
                        )
                    else:
                        st.write(dfRO.head())

                    # RoBERTa compound mean
                    roberta_mean = dfRO["roberta_compound"].mean()
                    roberta_mean_list = roberta_mean.tolist()
                    roberta_mean_slider = st.slider('-1.00: Negative, 0.00: Neutral, 1.00: Positive', min_value=-1.00, max_value=1.00, value=roberta_mean_list)
                    st.write('Total Sentiment: ', roberta_mean_slider)

                    # pie chart
                    roberta_pie_labels = 'Negative', 'Neutral', 'Positive'
                    roberta_neg_lines = dfRO["roberta_neg"].sum()
                    roberta_neu_lines = dfRO["roberta_neu"].sum()
                    roberta_pos_lines = dfRO["roberta_pos"].sum()
                    roberta_lines_sum = roberta_neg_lines+roberta_neu_lines+roberta_pos_lines
                    roberta_neg_perc = roberta_neg_lines/roberta_lines_sum
                    roberta_neu_perc = roberta_neu_lines/roberta_lines_sum
                    roberta_pos_perc = roberta_pos_lines/roberta_lines_sum
                    sizes = [roberta_neg_perc, roberta_neu_perc, roberta_pos_perc]
                    roberta_fig, roberta_ax = plt.subplots()
                    roberta_ax.pie(sizes, labels=roberta_pie_labels, autopct='%1.1f%%',
                            shadow=True, startangle=90)
                    roberta_ax.axis('equal')
                    st.pyplot(roberta_fig)
                    
                    #results
                    roberta_mean_str = str(roberta_mean)
                    roberta_cut_str = roberta_mean_str[:6]
                    st.warning("This is not a Financial Advisor")
                    if roberta_mean <= -0.5:
                        st.subheader("Your Final Sentiment Is: " + roberta_cut_str)
                        st.subheader("It's very likely that the company value will soon been DECREASING at a rapid rate!!! Be Careful!")
                    elif roberta_mean > -0.5 and roberta_mean < 0:
                        st.subheader("Your Final Sentiment Is: " + roberta_cut_str)
                        st.subheader("The near future of the company does not look too bright. Be Cautious")
                    elif roberta_mean > 0 and roberta_mean < 0.5:
                        st.subheader("Your Final Sentiment Is: " + roberta_cut_str)
                        st.subheader("The company will have a steady growth in the near future!")
                    else: 
                        st.subheader("Your Final Sentiment Is: " + roberta_cut_str)
                        st.subheader("The company's value is going to SKYROCKET very soon!")

            if analyser == 'VADER: Accurate & Fast':
                vader_analyser()
            elif analyser == 'RoBERTa: Premium Accuracy & Very Slow':
                roberta_analyser()

    elif searched_status == False and analyser == True:
        st.write("Search First!")
    elif searched_status == True and analyser == False:
        st.warning("Start the data!")
    elif searched_status == False and analyser == False:
        st.write("")


if ('Twitter' in selected_apis) == True:
    twitter()
elif ('Another Platform' in selected_apis) == True:
    st.write("Another Platform")
else:
    st.write("No APIs selected")