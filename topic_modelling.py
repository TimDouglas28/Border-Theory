"""
#######################################################################
#######   Generate BERTopic Topics for Unprocessed Twitter Data   #####
#######################################################################
"""
import pandas as pd
import numpy as np
import re
import nltk
from pathlib import Path
from bertopic import BERTopic
from collections import defaultdict
from textblob import TextBlob
from utils_calculations.utils import data_preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from utils_calculations.utils import read_pickle_dic, write_pickle_dic
from sentence_transformers import SentenceTransformer
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import IncrementalPCA
from bertopic.vectorizers import OnlineCountVectorizer

class generate_topics:
    """
    This is a class for generating BERTopic topics. To generate BERTopic topics for new Twitter data, 
        you must work through the following steps:
        1. Filter users: Filter out users who tweet >= a specified threshold value
        1. Get active members: Extract data for users who tweet in each year of the study period
        2. Filter active members: Filter active members based on whether they self-declare a profession
                in their description period. We obtain tweets for self-delcared profesionals as output.
        3. Get tokens: Tokenize all tweets for self-declared professionals.
        4. Extract nouns: Extract nouns from the tokenized tweet data for self-declared professionals
        5. Clean data: Clean the resulting data before passing it to BERTopic().
        6. Run BERTopic model
        7. Save BERTopic model

    You may also wish to run the following methods:
        Op_1: precompute_embeddings: This method precomputes tweet embeddings to shorten runtime of BERTopic modelling.
    """
    def __init__(self, community, _dir):
        """
        The constructor for the generate_topics class
        
        Parameters
        ----------
            community (str): The name of the Twitter community (without hashtag)
            data_file (str): File location of the unprocessed Twitter community data 
                            (e.g. '/Users/timdouglas/Desktop/MPhil:PhD CompSci @ UCL/css_proj/')
        """
        self.community = community
        self._dir = _dir

    def filter_users(self, threshold):
        """
        For a given community, read the raw Twitter data into a Pandas
        dataframe and filter out users who tweets >= some threshold value
        throughout the entire period of the tweets.
 
        Parameters
        ----------
            threshold (int): A threshold to filter out users (e.g, 2,000)
            
        Returns
        -------
            None
        """

        df = pd.read_csv(self._dir+'data/csvs/{}.csv'.format(self.community), low_memory=False)
        count = df[['author_id', 'tweet']].groupby(['author_id']).agg({'tweet': ['count']})
        count_thresh = count[(count[('tweet', 'count')] >= threshold)] 
        ids_list = list(count_thresh.index)
        df1 = df[df['author_id'].isin(ids_list)]
        df1.to_csv(self._dir+'data/csvs/unprocessed_tweets_{}_filt.csv'.format(self.community))

        return None

    def get_active_members(self):
        """
        Sample 'unprocessed_tweets_[community]_filt.csv' for users who
        tweet in each year of the period under study. We define active 
        members as users who have Tweeted in all of 2019, 2020, 2021, 2022.

        Parameters
        ----------
            None
            
        Returns
        -------
            None
        """

        df = pd.read_csv(self._dir+'data/csvs/unprocessed_tweets_{}_filt.csv'.format(self.community))
        # 1. Drop nans from created_at_year column.
        df = df.dropna(subset=['created_at_year'])
        # 2. Clean the data
        df = df.drop(df.loc[df['created_at_year']=='created_at_year'].index)
        df['created_at_year'] = df['created_at_year'].astype(int)
        df['author_id'] = df['author_id'].astype(str)
        
        # 3. Extract active members
        df_active_members = df[df.groupby('author_id')['created_at_year'].
        transform(lambda x : {2019, 2020, 2021, 2022}.issubset(set(x)))].sort_values(['author_id', 'created_at_year'])
        df_active_members.to_csv(self._dir+'data/csvs/unprocessed_{}_tweets_active_members.csv'.format(self.community))

        return df_active_members

    def filter_active_members(self, members_df, filter_list):
        """
        Filter 'unprocessed_[community]_tweets_active_members.csv' based on 
        whether users self-declare a profession in their description period.
        We determine user profession from linguistic substring matching.
        Specifically, we tokenize the description fiels and look for token
        matches to keywords in a separate list (filter list)

        Parameters
        ----------
            members_df (pandas.core.frame.DataFrame): Output of get_active_members() method.
            filter_list (list): A list of strings used to check for token matches in members_df 
                                description field

        Returns
        -------
            filtered_members (pandas.core.frame.DataFrame): DataFrame with users who 
                self-declare their profession.
        """

        tt = TweetTokenizer()
        desc_dict = {}
        desc = dict(zip(members_df['author_id'], members_df['description']))
        desc = {k: str(v) for k,v in desc.items()}
        for _id, d in desc.items():
            token_desc = tt.tokenize(d)
            desc_dict.update({_id: token_desc})
        teacher_keys = [key for key, value in desc_dict.items() for x in filter_list if x in value]
        # teacher_keys = [int(x) for x in teacher_keys]
        filtered_members = members_df[members_df['author_id'].isin(teacher_keys)]
        filtered_members.to_csv(self._dir+'data/csvs/unprocessed_{}_tweets_active_members*.csv'.format(self.community))

        return filtered_members

    def normalization(self, tweet_list):
        """
        A function that performs Lexicon Normalization to a list
        of tweets. This is done to remove any multiple representations of 
        the same word (e.g. play and playing)

        Parameters
        ----------
            tweet_list (list): A list of tweets

        Returns
        -------
            _norm (list): A list of nromalized tweets
        """

        lem = WordNetLemmatizer()
        verb_norm = [lem.lemmatize(word, "v") for word in tweet_list]
        adj_norm = [lem.lemmatize(word, "a") for word in verb_norm]
        noun_norm = [lem.lemmatize(word, "n") for word in adj_norm]
        _norm = [lem.lemmatize(word, "r") for word in noun_norm]

        return _norm

    def form_sentence(self, tweet):
        """
        A function that leverages textblob to process text
        (e.g., a tweet)

        Parameters
        ----------
            tweet(str): A tweet string

        Returns
        -------
            Processed tweet string
        """
        tweet_blob = TextBlob(tweet)

        return ' '.join(tweet_blob.words)
    
    def data_preprocessing(self, df):
        """
        A function that leverages nltk and regex libraries to preprocess raw
        tweet data.

        Parameters
        ----------
            df (pandas.core.frame.DataFrame):a DataFrame object containing the tweets 
                (e.g., 'unprocessed_[community]_tweets_active_members*.csv')

        Returns
        -------
            full_text (list): A list containing the pro-processed tokens
        """

        # 1. Create a list from the tweet field in dataframe
        text = df['tweet'].values 

        # 2. Create an empty list for storing tokens
        full_text = []
        tt = TweetTokenizer()

        # 3. Iterate through the list of tweets, cleaning each tweet in turn
        for tweet in range(len(text)):
            token = tt.tokenize(text[tweet])
            norm = self.normalization(token)
            new_tokens = [word.lower() for word in norm if word not in stopwords.words('english')]
            untagged_tokens = [word for word in new_tokens if "#" not in word]
            userless_tokens = [word for word in untagged_tokens if "@" not in word]
            blobbed_tokens = [self.form_sentence(word) for word in userless_tokens]
            clean_tokens = [re.sub(r'[^\w\s]', '', word) for word in blobbed_tokens if word != '']
            cleaner_tokens = [word for word in clean_tokens if "https" not in word]
            cleanest_tokens = [word for word in cleaner_tokens if word != "rt"]
            c_tokens = [re.sub(r'[^a-zA-Z]', '', word) for word in cleanest_tokens]
            remove_numbers = [re.sub(r'\d+', '', word) for word in c_tokens]
            remove_whitespace = [word.replace(' ', '') for word in remove_numbers]
            _tokens = [word for word in remove_whitespace if len(word)>2]
            cleaned_tokens = self.normalization(_tokens)
            final_tokens = [word for word in cleaned_tokens if word] #removes empty strings
            full_text.append(final_tokens)
    
        return full_text
    
    def get_tokens(self):
        """
        Tokenize all tweets for self-delcared professionals in 
        'unprocessed_[community]_tweets_active_members*.csv'. 

        Parameters
        ----------
            None

        Returns
        -------
            activemembers_df (pandas.core.frame.DataFrame): a DataFrame object 
                containing tweets for self-declared professionals
                (e.g., 'unprocessed_[community]_tweets_active_members*.csv')
        """

        # 1. Read in the tweets for self-declared professionals as a csv.
        activemembers_df = pd.read_csv(
            self._dir+'data/csvs/unprocessed_{}_tweets_active_members*.csv'.format(self.community), 
            usecols=['author_id', 'tweet_id', 'tweet', 'created_at'] 
        ) 

        # 2. Tokenize tweets by calling data_preprocessing method.
        tokens_list = self.data_preprocessing(activemembers_df)

        # 3. Store the tokenized data. 
        # Create a [community] path if it exists, else do nothing
        Path(self._dir + 'data/pickles/study/{}/BERTopic_input_data'.format(self.community)).mkdir(parents=True, exist_ok=True)
        write_pickle_dic(tokens_list, self._dir + 'data/pickles/study/{}/BERTopic_input_data/tokens_list'.format(self.community))
        
        return activemembers_df
    
    def extract_nouns(self, activemembers_df):
        """
        Extract nouns from the tokenized tweet data for self-declared
        professionals. NB: This is not a requirement for BERTopic, however
        we discovered better performance (qualitatively in terms on most 
        probable words in topics) when extracting nouns.

        Parameters
        ----------
            activemembers_df (DataFrame): Tweets for self-declared professionals (output of get_tokens() method).

        Returns
        -------
            None
        """

        # 1. Read in the pickle file containing the tokenized tweets and add toa column in activemembers_df
        tokens_list = read_pickle_dic(self._dir + 'data/pickles/study/{}/BERTopic_input_data/tokens_list'.format(self.community))
        activemembers_df['token_list'] = tokens_list
        activemembers_df['token_list'].replace([], np.nan, inplace=True)
        activemembers_df.dropna(subset=['token_list'], inplace=True)
        activemembers_df['tweet_id'] = activemembers_df.tweet_id.astype(int)
        tokens_dic = dict(zip(activemembers_df.tweet_id, activemembers_df.token_list))

        # 2. Extract nouns from the tweets
        pos_dic = {
            tweet_id: nltk.pos_tag(tlist) for tweet_id, tlist in tokens_dic.items()
        }
        nouns = defaultdict(list)
        for num, tuple_list in pos_dic.items():
            for i in range(len(tuple_list)):
                k, l = tuple_list[i] 
                # NB: Only nouns considered. Add/Remove as required
                if l =='NN' or l == 'NNS':
                    nouns[num].append(k)
        final_nouns = {
            num: tlist for num, tlist in nouns.items()
        }
        final_nouns_cleaned = {tweet_id: ' '.join(tokens) for tweet_id, tokens in final_nouns.items()}
        cleaned_dic = {int(k):v for k, v in final_nouns_cleaned.items()}
        activemembers_df['token_nouns'] = activemembers_df.tweet_id.map(cleaned_dic)
        activemembers_df.dropna(subset=['token_nouns'], inplace=True)

        # 3. Save result
        Path(self._dir + 'data/csvs/BERTopic/{}'.format(self.community)).mkdir(parents=True, exist_ok=True)
        activemembers_df.to_csv(self._dir + 'data/csvs/BERTopic/{}/{}_nouns_truncated.csv'.format(self.community, self.community))

        return None

    def process_data(self):
        """
        Clean and reformat the data before passing it to BERTopic() 

        Parameters
        ----------
            activemembers_df (DataFrame): Tweets for self-declared professionals (output of get_tokens() method).

        Returns
        -------
            None
        """

        # 1. Read in the extracted nouns csv as a Pandas DataFrame.
        df = pd.read_csv(
            self._dir+'data/csvs/BERTopic/{}/{}_nouns_truncated.csv'.format(self.community, self.community),
            usecols=['author_id', 'tweet_id', 'created_at', 'token_nouns'],
        ) 

        # 2. Clean and specify timestamps and tweets.
        df = df.drop_duplicates(subset=['token_nouns'])
        df['created_at_date'] = pd.to_datetime(df['created_at'])
        df['created_at_date'] = df['created_at_date'].dt.strftime('%Y-%m-%d')
        df['tweet'] = df['token_nouns'] # 
        df.dropna(subset=['tweet'], inplace=True) # complained df still have nan vals so rewrite this line
        df.tweet_id = df.tweet_id.astype(int)
        df.tweet = df.apply(lambda row: re.sub(r"http\S+", "", row.tweet).lower(), 1)
        df.tweet = df.apply(lambda row: " ".join(filter(lambda x:x[0]!="@", row.tweet.split())), 1)
        df.tweet = df.apply(lambda row: " ".join(re.sub("[^a-zA-Z]+", " ", row.tweet).split()), 1)
        timestamps = df.created_at_date.to_list()
        tweets = df.tweet.to_list()

        return tweets, timestamps

    def precompute_embeddings(self, tweets):
        """
        5.2: Pre-compute embeddings to save time. Typically, we want to iterate 
        fast over different versions of our BERTopic model whilst we are trying to 
        optimize it to a specific use case. To speed up this process, we can pre-compute the embeddings, 
        save them, and pass them to BERTopic so it does not need to calculate the embeddings each time.

        Parameters
        ----------
            tweets (list): A list of tweets as strings.

        Returns
        -------
            embeddings: The resulting embeddings file
        """

        embedding_model = SentenceTransformer('all-MiniLM-L6-v2') # default transformer model for language='english'. Maybe try out several and see which works best? 
        embeddings = embedding_model.encode(tweets, show_progress_bar=False)

        return embeddings
    
    def compute_online_bertopic(self, tweets, nclusters):
        """ 
        Compute online topic modelling on the processed tweets for self-declared
        professionals. We do this because offline topic modelling fails to run if 
        the input data is too large. To avoid memory issues, we train the bertopic
        classifier in an online setting.

        Parameters
        ----------
            tweets (list): A list of the tweets forself-declared professionals.

        Returns
        -------
            topic_model (bertopic._bertopic.BERTopic): BERTopic model
            topics (list): A list of all row-level topic mappings to tweets in 'tweets'
            tweets (list): A list of all the tweets used to generate topics
            timestamps (list): A list of row-level %Y-%m-%d timestamp mappings to tweets in 'tweets'
        """

        # 1. Remove any stopwords from tweets.
        from nltk.corpus import stopwords
        tweets = [tweet for tweet in tweets if tweet not in stopwords.words('english')]

        # 2. Chunk data into N chunks
        N = 100
        doc_chunks = [tweets[i:i+N] for i in range(0, len(tweets), N)]
        doc_chunks = [[string for string in sublist if string] for sublist in doc_chunks]
        stopwords = list(stopwords.words('english')) + ['http', 'https', 'hashtag', 'retweet']

        # 3. Prepare sub-models that support online learning
        umap_model = IncrementalPCA(n_components=5)
        cluster_model = MiniBatchKMeans(n_clusters=nclusters, random_state=0)
        vectorizer_model = OnlineCountVectorizer(stop_words=stopwords, delete_min_df=2)
        sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = sentence_model.encode(tweets, show_progress_bar=True) 

        # 4. Instantiate BERTopic model
        topic_model = BERTopic(umap_model=umap_model,
                            hdbscan_model=cluster_model,
                            vectorizer_model=vectorizer_model,
                            top_n_words = 10,
                            n_gram_range=(1,3),
                            verbose=True,
                            embedding_model=sentence_model,
                            diversity=.1,
                            calculate_probabilities=True)

        # 5. Incrementally fit the topic model by training on N documents at a time and track the topics in each iteration
        embeddings_store = []
        topics = []
        for docs in doc_chunks:
            embeddings = sentence_model.encode(docs, show_progress_bar=True) 
            topic_model.partial_fit(docs, embeddings)
            topics.extend(topic_model.topics_)
            embeddings_store.append(embeddings)

        topic_model.topics_ = topics

        return topic_model, topics, tweets, embeddings_store

    def save_topic_model(self, topic_model, tweets, topics, timestamps, embeddings):
        """
        Save a BERTopic model, and its associated tweets, topics and timestamps (for temporal plotting)
        
        Parameters
        ----------
            tweets (list): A list of tweets as strings.
            embeddings (array): An array of word embeddings for the given community

        Returns
        -------
            None
        """
        Path(self._dir + '/BERTopic/{}'.format(self.community)).mkdir(parents=True, exist_ok=True)
        topic_model.save(self._dir + '/BERTopic/{}/{}_nouns_model'.format(self.community, self.community))
        write_pickle_dic(tweets, self._dir + '/BERTopic/{}/{}_tweets'.format(self.community, self.community))
        write_pickle_dic(topics, self._dir + '/BERTopic/{}/{}_topics'.format(self.community, self.community))
        write_pickle_dic(timestamps, self._dir + '/BERTopic/{}/{}_timestamps'.format(self.community, self.community))
        write_pickle_dic(embeddings, self._dir + '/BERTopic/{}/{}_embeddings'.format(self.community, self.community))

        return None

"""
Instantiate generate_topics class and work through the following steps to obtain
topics for community data below.
"""

# 0. Initialise instance variables
community, _dir = 'Journalists', '/Users/timdouglas/Desktop/MPhil:PhD CompSci @ UCL/css_proj/' 
# ScottishTeachers # IrishTeachers # Journalists

# 1. Instantiate generate_topics class
bert = generate_topics(community, _dir)

# 2. Filter users based on total number of tweets from 2019 to 2022
bert.filter_users(threshold=2000)

# 3. Get active members 
members_df = bert.get_active_members()

# # 4. Determine professionals from dataset
filter_list = read_pickle_dic('{}_filter_list'.format(community))
filtered_members = bert.filter_active_members(members_df, filter_list)

# 5. Tokenize filterd self-declared professionals:
activemembers_df = bert.get_tokens()

# 6. Extract nouns from tokenised data to improve topic interpretability
bert.extract_nouns(activemembers_df)

# 7. Clean and reformat the data before passing it to BERTopic()
tweets, timestamps = bert.process_data()

# 8. Compute online BERTopic model
nclusters = 5 # NB: Consider looping through multiple cluster values and comparing topics
topic_model, topics, filtered_tweets, embeddings = bert.compute_online_bertopic(tweets, nclusters=nclusters)

# 9. Save BERTopic Topics
bert.save_topic_model(topic_model, filtered_tweets, topics, timestamps, embeddings)

debug2 = 0
