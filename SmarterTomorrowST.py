import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
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

header = st.container()
with header: 
    st.title('SmarterTomorrow')
    st.caption('**Description**: ')
    st.caption("Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.")

#Search
st.header("Search")
search_form = st.form("search_form")
with search_form: 
    query = st.text_input("Company Name (aka Twitter Tag)", placeholder='e.g. Google, Tesla')
    # duration = st.selectbox("Choose a duration", ('Recent One Quater', 'Recent Two Quaters', 'Recent Year', 'Recent Two Year'))
    selected_apis = st.multiselect('Select APIs', ['Twitter', 'Another Platform'], default='Twitter')
    analyser = st.selectbox("Choose an Analyser", ('VANDER: Accurate & Fast', 'RoBERTa: Premium Accuracy & Very Slow'))
    checkbox_val = st.checkbox("I agree to the Terms & Conditions, and that this is not a Financial Advisor!")
    searched = st.form_submit_button("Search")
    if searched:
        if len(query) != 0 and checkbox_val == True:
            # st.write("Company Name: ", query)
            # st.write("Duration: ", duration,)
            # st.write("Analyser: ", analyser)
            # st.write("Checkbox: ", checkbox_val)
            st.write("Successful!")
        elif len(query) == 0 and checkbox_val == True:
            st.warning("**Please Enter a Company Name**")
        elif len(query) != 0 and checkbox_val == False:
            st.warning("**Please agree to the Terms & Conditions!**")
        elif len(query) == 0 and checkbox_val == False:
            st.warning("**No Company Input, and agree to the Terms & Conditions! Try again!**")

### Another Platform

### Twitter
def twitter():
    #Twitter API input
    twitter_section = st.header("Twitter")
    with twitter_section:
        df = pd.read_csv('output.csv')
            # cleanup
        df = df.drop(['Unnamed: 0'], axis=1)

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
            for i in df["tweet_text"]:
                sr = pd.Series(i)
                sr.to_string()
                tokenizer = RegexpTokenizer(r'\w+')
                tokens = tokenizer.tokenize(i)
            fdist = FreqDist()
            for word in tokens:
                fdist[word.lower()] += 1
            st.subheader("The 10 most common words")
            most_common10 = fdist.most_common(10)
            most_common10DF = pd.DataFrame(most_common10, columns = ['Words', 'Frequency'])
            all_fdist = pd.Series(dict(most_common10))
            fig, ax = plt.subplots(figsize=(10,10))
            all_plot = sns.barplot(x=all_fdist.index, y=all_fdist.values, ax=ax)
            plt.xlabel('Words')
            plt.ylabel('Frequency')
            st.pyplot(fig)

        # Sentiment
        sentiment = st.container()
        with sentiment: 
            st.header("Sentiment Analysis")
            
            ## Analysers
                # Vader
            def vender_analyser():
                st.subheader("VANDER: Socia Media Dedicated")
                if st.checkbox("Start Analysing!"): 
                    analyzer = SentimentIntensityAnalyzer()

                    # First back up the values in 'dfV'
                    dfV = pd.DataFrame(df, columns = ['username', 'tweet_id', 'tweet_text', 'vader_neg', 'vader_neu', 'vader_pos', 'vader_compound'])
                    dfV['vader_neg'] = df['tweet_text'].apply(lambda x:analyzer.polarity_scores(x)['neg'])
                    dfV['vader_neu'] = df['tweet_text'].apply(lambda x:analyzer.polarity_scores(x)['neu'])
                    dfV['vader_pos'] = df['tweet_text'].apply(lambda x:analyzer.polarity_scores(x)['pos'])
                    dfV['vader_compound'] = df['tweet_text'].apply(lambda x:analyzer.polarity_scores(x)['compound'])

                    if st.checkbox('Show full data table with VANDER values'):
                        st.subheader('Full Table With VANDER')
                        st.write(dfV.head())
                        if st.button('Download Full Data CSV'):
                            dfV.to_csv('twitterVander.csv', sep='\t', encoding='utf-8')


                    # VADER only
                    st.subheader('VANDER Table')
                    dfVO = pd.DataFrame(dfV, columns=['vader_compound', 'vader_neg', 'vader_neu', 'vader_pos', 'tweet_text'])
                    st.write(dfVO)

                    # mean
                    vader_mean = dfVO["vader_compound"].mean()
                    vader_mean_slider = st.slider(
                        '-1.00: Negative, 0.00: Neutral, 1.00: Positive',
                        -1.00, 1.00, value=vader_mean)
                    st.write('Total Sentiment: ', vader_mean_slider)
                    
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
                        st.subheader("The company's value is going to SKYROCKET very soon!!!")
            
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

                    # First back up the values in 'dfR'
                    if st.checkbox('Show full data table with RoBERTa values'):
                        st.subheader('Full Table With RoBERTa')
                        st.write(dfR.head())
                        if st.button('Download Full Data CSV'):
                            dfR.to_csv('twitterRoBERTa.csv', sep='\t', encoding='utf-8')

                    #RoBERTa only
                    dfRO = pd.DataFrame(dfR, columns = ['roberta_compound', 'roberta_neg', 'roberta_neu', 'roberta_pos', 'tweet_text'])
                    st.write(dfRO.head())

                    # # RoBERTa compound mean
                    # roberta_mean = dfRO["roberta_compound"].mean()
                    # roberta_mean_slider = st.slider('-1.00: Negative, 0.00: Neutral, 1.00: Positive', min_value=-1.00, max_value=1.00, value=roberta_mean)
                    # st.write('Total Sentiment: ', roberta_mean_slider)
                    
                    # #results
                    # roberta_mean_str = str(roberta_mean)
                    # roberta_cut_str = roberta_mean_str[:6]
                    # st.warning("This is not a Financial Advisor")
                    # if roberta_mean <= -0.5:
                    #     st.subheader("Your Final Sentiment Is: " + roberta_cut_str)
                    #     st.subheader("It's very likely that the company value will soon been DECREASING at a rapid rate!!! Be Careful!")
                    # elif roberta_mean > -0.5 and roberta_mean < 0:
                    #     st.subheader("Your Final Sentiment Is: " + roberta_cut_str)
                    #     st.subheader("The near future of the company does not look too bright. Be Cautious")
                    # elif roberta_mean > 0 and roberta_mean < 0.5:
                    #     st.subheader("Your Final Sentiment Is: " + roberta_cut_str)
                    #     st.subheader("The company will have a steady growth in the near future!")
                    # else: 
                    #     st.subheader("Your Final Sentiment Is: " + roberta_cut_str)
                    #     st.subheader("The company's value is going to SKYROCKET very soon!!!")

            if analyser == 'VANDER: Accurate & Fast':
                vender_analyser()
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