"""
#######################################################################
#######   Work-Life Tweet Activity Analysis of BERTopic Topics    #####
#######################################################################
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import seaborn as sns
import scipy.stats as stats
import re
import pickle
import textwrap
from datetime import datetime
from numpy.linalg import norm
from pydoc_data.topics import topics
from pyexpat import model
from bertopic import BERTopic
from functools import reduce
from numpy import mean
from collections import defaultdict
from octis.evaluation_metrics.diversity_metrics import TopicDiversity
from octis.evaluation_metrics.topic_significance_metrics import KL_uniform
from nltk.sentiment.vader import SentimentIntensityAnalyzer

class work_life_analysis:
    """
    This is a class for analysing tweets assigned to topics created using
    a BERTopic topic model. You must work through the following steps:
        1. Load in a topic model (from constructor)
        2. Map topics: Map BERTopic topics back to the original tweets dataset.
        3. Transform data: Transform the data to extract work and life tweets
        4. Compute work-life tweet frequency analysis:
            4.1: Get crisis information: Get the information required to run the analysis
                on a specific lockdown period (e.g., months and years).
            4.2: Get work-life tweet frequency: Get results for the work-life tweet frequency over time
                (e.g., per lockdown).
            4.3: Get subjective well-being: Get results for the subjective well-being over time.
            4.4: Plots the results for the work-life analysis and subjective well-being

    You may also wish to run the following methods:
        Op_1. evaluate_model: This method provides metrics to quantitatively evaluate a BERTopic model
        Op_2. run_statistical_test: This method computes a test (Wilcoxon-Mann-Whitney and Kolmogorov-Smirnov) 
            for the work frequency vectors and life frequency vectors for each day across periods.
    """

    def __init__(self, _dir, community, bertmodel, tweets, topics, timestamps, embeddings):
        """
        The constructor for the generate_topics class.
        
        Parameters
        ----------
            _dir (str): Directory location of the unprocessed Twitter community data 
                        (e.g. '/Users/timdouglas/Desktop/MPhil:PhD CompSci @ UCL/css_proj/')
            community (str): The community you wish to analyse (e.g. 'phd')
            bertmodel (bertopic._bertopic.BERTopic): BERTopic model
            tweets (list): A list of all the tweets used to generate topics
            topics (list): A list of all row-level topic mappings to tweets in 'tweets'
            timestamps (list): A list of row-level %Y-%m-%d timestamp mappings to tweets in 'tweets'
        """

        self._dir = _dir
        self.community = community
        self.bertmodel = bertmodel
        self.tweets = tweets
        self.topics = topics
        self.timestamps = timestamps
        self.embeddings = embeddings
    
    def evaluate_model(self, bertmodel):
        """
        Evaluate a fitted BERTopic model using topic diversity 
        and topic significance metrics.

        Parameters
        ----------
            bertmodel (bertopic._bertopic.BERTopic): BERTopic model

        Returns
        -------
            topic_diversity_core (float): Topic diveristy score
            topic_significance_score (float): Topic significance score
        """

        #---- Topic Diversity Calculation ----

        # reformat data to get required output
        tops = []
        for i in range(len(bertmodel.get_topics())):
            t = [x[0] for x in bertmodel.get_topic(i)]
            tops.append(t)

        topic_diversity_dic = {'topics': tops}

        # Initialize metric
        topic_diversity = TopicDiversity(topk=len(bertmodel.get_topic(0)))

        # Retrieve metrics score
        topic_diversity_score = topic_diversity.score(topic_diversity_dic)
        # print("Topic diversity: "+str(topic_diversity_score))

        #---- Topic Significance Calculation ---- 

        # reformat data to get required output
        topic_word_matrix = bertmodel.c_tf_idf_.toarray()
        topic_word_matrix = np.delete(topic_word_matrix, obj=0, axis=0)
        topic_significance_dic = {'topic-word-matrix': topic_word_matrix}

        # Initialize metric
        topic_significance = KL_uniform()

        # Retrieve metrics score
        topic_significance_score = topic_significance.score(topic_significance_dic)
        # print("Topic significance: "+str(topic_significance_score))      

        return topic_diversity_score, topic_significance_score
            
    def map_topics(self):
        """
        Fora given BERTopic model, map topics back to the original tweet data.
        The original tweet data contains the processed tweets for all self-declared
        professionals (e.g., all Journalists).

        Parameters
        ----------
            None
        
        Returns
        -------
            orig_merge (pandas.core.frame.DataFrame): A Dataframe object that maps BERTopic output back to 
                original tweet data (the processed tweets for self-delcared professionals)
        """

        # 1. Read the original tweet data into a Pandas dataframe
        orig = pd.read_csv(self._dir+'unprocessed_{}_tweets_active_members*.csv'.format(self.community), 
            usecols=['author_id', 'tweet_id', 'tweet', 'created_at']) # 'location'

        # 2. Create a DataFrame object mapping tweets to their respective topics, for all required topics
        max_topics = len(self.bertmodel.get_topics()) # NB: this should equal the number of topics you want to analyse
        tops = [i for i in range(max_topics)]
        topic_doc = pd.DataFrame({'Topic': self.topics, 'token_nouns': self.tweets})
        topic_doc = topic_doc.loc[topic_doc['Topic'].isin(tops), :]

        # 3. Read the transformed tweet data in as a dataframe (all filtered tweets containing nouns)
        transformed = pd.read_csv(self._dir+'{}/{}_nouns_truncated.csv'.format(self.community, self.community),
            usecols=['author_id', 'tweet_id', 'created_at', 'token_nouns'], lineterminator='\n'
        )

        # 4. Merge topic_doc with transformed to get the tweet_id
        nouns_merge = pd.merge(topic_doc, transformed, on='token_nouns')

        # 5. Merge nouns_merge with the original dataset, orig. We merge on the tweet_id column, 
        # which appears in both nouns_merge and orig
        orig_merge = pd.merge(nouns_merge, orig, on='tweet_id')
        orig_merge = orig_merge.drop_duplicates(subset=['tweet'])

        # 6. Rename columns in orig_merge and drop duplicate columns
        orig_merge = orig_merge.rename(columns={'author_id_x': 'author_id', 'created_at_x': 'created_at'})
        orig_merge = orig_merge.drop(['author_id_y', 'created_at_y'], axis=1)

        return orig_merge

    def transform_data(self, orig_merge, work_tops, life_tops):
        """
        Extract work and life tweets from the topics data.
        
        Parameters
        ----------
            orig_merge (pandas.core.frame.DataFrame): A Dataframe object that maps BERTopic output back to original data
            work_tops (list): A list of integers corresponding to topics labelled as 'work', e.g., [2,3,7]
            life_tops (list): A list of integers corresponding to topics labelled as 'life', e.g., [0,1,4,5,6,8,9]

        Returns
        -------
            result (list): A list of Dataframes, where each DataFrame is all of a user's w and l tweets
        """

        # 0. Rename 'orig_merge' to 'df' for convenience
        df = orig_merge

        # 1. Clean tweets in preparation for sentiment analysis later on
        df['tweet'] = df.apply(lambda row: re.sub(r"http\S+", "", row.tweet).lower(), 1)
        df['tweet'] = df.apply(lambda row: " ".join(filter(lambda x:x[0]!="@", row.tweet.split())), 1)
        df['tweet'] = df.apply(lambda row: " ".join(re.sub("[^a-zA-Z]+", " ", row.tweet).split()), 1)

        # 2. Separate work and life tweets
        w = work_tops
        l = life_tops

        # 3. Label work and life tweets as 'w' and 'l' respectively in orig_merge
        w_chunk = df[df['Topic'].isin(w)]
        w_chunk['label'] = 'w'
        l_chunk = df[df['Topic'].isin(l)]
        l_chunk['label'] = 'l'
        
        # 4. Concatenate work and life tweets
        c = pd.concat([w_chunk, l_chunk]) # NB: Will not equal len(df) if only a subset of topics selected from df are in w_chunk and l_chunk

        # 5. Pull out required columns for analysis
        c_filtered = c[["Topic", "tweet", "label", "author_id", "created_at"]] # "location", "local_time"
        c_list = [d for _, d in c_filtered.groupby(['author_id'])]

        # 6. Sort c_list by timestamp
        result = [df.sort_values(by='created_at') for df in c_list]

        return result

    def compute_work_life_frequency(self, user, result, year, month, day, thresholds, inc_topics, _continuum):
        """ 
        *-*-* SINGLE USER ANALYSIS *-*-*: For a given user, compute boxplot and baricenter 
        of work and life tweets for each day in a period (e.g., a steady state, covid, etc.)

        Parameters
        ----------
            user (int): An integer which we use to index result to get data for a single user
            result (list): A list of Dataframes, where each DataFrame is all of a user's w and l tweets
            year (list): A list of an int corresponding to a single year (e.e., [2019])
            month (list): A list of strings corresponding to a named month/months (e.g., ['February', 'March', 'April'])
            day (str): A str corresponding to a named day (e.g., 'Monday')
            thresholds (list): A list of strings correspnding to cutoff periods (e.g., ['2019-02-04', '2019-04-28'])
            *NEW: Added 11/12/22* _continuum (bool): If True, does not truncate dates based on threshold values
            (thresholds kwarg still needs to be passed, but is ignored).

        Returns
        -------
            single_user (dict): A dictionary of work and life tweet times (in seconds) for a given user, for all days in a period
            result (list): A list of work and life tweet times, updated with DBA result
        """
        
        # 0. Pull data for a given user by indexing result with user
        df = result

        # 1. Create time columns for each DataFrame in result (list)
        df['timestamp'] = pd.to_datetime(df['created_at'], format='%Y-%m-%d %H:%M:%S')
        df['year'] = df['timestamp'].dt.isocalendar().year
        df['month'] = df['timestamp'].dt.month
        df['month_name'] = df['timestamp'].dt.month_name()
        df['week'] = df['timestamp'].dt.isocalendar().week
        df['day'] = df['timestamp'].dt.day
        df['day_name'] = df['timestamp'].dt.day_name()
        df['hour'] = df['timestamp'].dt.hour

        # 3.A. Define your steady state (ss) for data segmented by topic
        ss_tops = []
        for t in sorted(list(df['Topic'].unique())):
            top_df = df[df['Topic'] == t]
            ss = top_df[(top_df['year'].isin(year)) & (top_df['month_name'].isin(month))]
            if _continuum:
                ss = ss[ss['day_name']==day]
            else:
                ss = ss[(~(ss['timestamp'] < thresholds[0])) & (~(ss['timestamp'] > thresholds[1]))]
                ss = ss[ss['day_name']==day]
            ss['time'] = pd.to_datetime(ss['timestamp']).dt.time # 'timestamp' # 'local_time'
            ss['time'] = ss['time'].astype('str') 
            #ss_df = pd.DataFrame(ss['time'])
            ss_df = pd.DataFrame(ss[['time', 'tweet', 'author_id']])
            ss_df['{}'.format(t)] = pd.to_timedelta(ss_df['time']).dt.total_seconds()
            ss_df = ss_df.drop(['time'], axis=1)
            ss_tops.append(ss_df)

        # 3.B. Define your steady state (ss) for data segmented by w and l
        ss_labels = []
        ss_labels_tweets = []
        for label in ['w', 'l']:
            ss = df[(df['year'].isin(year)) & (df['month_name'].isin(month))]
            if _continuum:
                ss = ss[ss['label']==label]
                ss = ss[ss['day_name']==day]
            else:
                ss = ss[(~(ss['timestamp'] < thresholds[0])) & (~(ss['timestamp'] > thresholds[1]))]
                ss = ss[ss['label']==label]
                ss = ss[ss['day_name']==day]
            ss['time'] = pd.to_datetime(ss['timestamp']).dt.time # 'timestamp' # 'local_time'
            ss['time'] = ss['time'].astype('str') 
            #ss_df = pd.DataFrame(ss['time'])
            ss_df = pd.DataFrame(ss[['time', 'tweet', 'author_id']])
            ss_df['time'] = pd.to_timedelta(ss_df['time']).dt.total_seconds()
            ss_labels.append(ss_df)

        # 4. Rename work, work_tweet, life and life_tweet columns in ss_labels
        ss_labels[0] = ss_labels[0].rename(columns={'time': 'work', 'tweet': 'work_tweet', 'author_id': 'work_author_id'})
        ss_labels[1] = ss_labels[1].rename(columns={'time': 'life', 'tweet': 'life_tweet', 'author_id': 'life_author_id'})

        # 5. Create dictionary with data from ss_tops and ss_labels
        # 5.A For daily tweet volume data calculated per topic:
        single_user_topics = ss_tops[0].to_dict(orient='list')
        for i in range(len(ss_tops)):
            single_user_topics.update(ss_tops[i].to_dict(orient='list'))

        if inc_topics:
            # 5.B1 For daily tweet volme data calculated per label
            single_user = ss_labels[0].to_dict(orient='list')
            single_user.update(ss_labels[1].to_dict(orient='list'))

            # 5.C Update single_user with single_user_topics to group labels and topics data together
            single_user.update(single_user_topics)
        else:
            # 5.B2 For daily tweet volme data calculated per label
            single_user = ss_labels[0].to_dict(orient='list')
            single_user.update(ss_labels[1].to_dict(orient='list'))

        return single_user

    def compare_work_life_frequency(self, weekly_results, week, thresholds, inc_topics, norm, flex, topic_labels, wd_vs_we):
        """ 
        Display a violin plot of work and life tweet activity for users. 

        Parameters
        ----------
            single_user (dictionary): A dictionary of labels/topics and their frequency values per day
            day (string): A string of a given day in the week (e.g., 'Monday')
            thresholds (list): A list of strings corresponding to beginning and end date of period
            user (int): An integer referring to the index of a specific user in 'result'. Used to index
                        'result' to obtain all tweets to plot for that user.
            periods1 (tuple): A tuple of information required for plotting - year, months, days, periods.
            labeldict2 (dict): A dictionary mapping BERTopic topic names (which are integers) to labels.
            inc_topics (bool): A boolean for specifying whether topic labels should be plotted.
                                If False, topics will be plotted without labels. If False in 
                                single_user_analysis() method, topics will not be plotted at all.

        Returns
        -------
            None
        """

        # Specify flag for including/excluding topic labels.
        if inc_topics:
            for period in range(len(weekly_results)):
                for day in range(len(weekly_results.get(period))):
                    for label in list(topic_labels):
                        weekly_results[period][day][topic_labels.get(label)] = \
                        weekly_results[period][day].pop(label)
        else:
            pass
        
        # Convert periods in threshold to names
        period_names = defaultdict(list)
        store = []
        for period in range(len(thresholds)):
            # first, convert period date to english name
            for date in range(len(thresholds[period])):
                d = datetime.strptime(thresholds[period][date], '%Y-%m-%d')
                d = d.strftime('%a %d %b %Y')
                period_names[period].append(d)
        thresholds = list(dict(period_names).values())

        periods = []
        for period in thresholds:
            p = period[0] + ' to ' + period[1]
            periods.append(p)

        for period in range(len(weekly_results)):
            for day in range(len(weekly_results.get(period))):
                for label in list(topic_labels):
                    keys = list(weekly_results[period][day].keys())

        for period in range(len(weekly_results)):
            for day in range(len(weekly_results.get(period))):
                for k in keys:
                    arrays = [weekly_results[period][day][k]]
                    max_length = 0
                    for array in arrays:
                        max_length = max(max_length, len(array))
                    for array in arrays:
                        array += [np.nan] * (max_length - len(array))

        comp_dic = defaultdict(lambda: defaultdict(lambda: defaultdict (list)))
        for k in keys:
            for period in range(len(weekly_results)):
                for day in range(len(weekly_results.get(period))):
                    comp_dic[day][k][periods[period]].append(weekly_results[period][day].get(k))

        final_comp_dic = {
            day: {
                k: {
                    period: [val for val in vals[0]] for period, vals in period_dic.items()
                } for k, period_dic in k_dic.items()
            } for day, k_dic in comp_dic.items()
        }
        
        daily_df_store = []
        for day, comp_dic in final_comp_dic.items():
            df = pd.DataFrame(columns=['label'])
            for period in periods:
                df[periods] = None 
            for key, value in comp_dic.items():
                for period in periods:
                    period_df = pd.DataFrame({'label': [key]*len(value[period]), period: value[period]})
                    df = df.append(period_df, ignore_index=True)
            daily_df_store.append(df)

        daily_results_store = []
        for d in range(len(daily_df_store)):
            df = pd.melt(frame = daily_df_store[d], id_vars = 'label', value_vars = periods, 
            var_name = 'type', value_name = 'value')
            try:
                df['value'] = df['value'].astype(float)
            except:
                pass
            daily_results_store.append(df)

        return daily_results_store, periods

    def plot_work_norm_frequency(self, comparison_results_final_list, community, wd_vs_we=True):
        """ 
        Plots the normalised frequency for work and life across all lockdown periods:
            'Mon 23 Mar 2020 to Mon 01 Jun 2020', 
            'Thu 05 Nov 2020 to Wed 02 Dec 2020', 
            'Wed 06 Jan 2021 to Mon 08 Mar 2021'

        Parameters
        ----------
           comparison_results_final_list (list): A list of dataframes containing the tweet frequency for each lockdown.

        Returns
        -------
            None
        """
        # 1. Ensure You concatenate all weekdays into one dataframe for comparison_results_final.
        freq_list = []
        for lockdown_result in comparison_results_final_list:
            comparison_results_final = lockdown_result
            wd_vs_we=True
            if wd_vs_we:
                if len(comparison_results_final) == 7:
                    wd = [pd.concat(comparison_results_final[0:4])]
                    we = comparison_results_final[5:7] 
                    comparison_results_final = wd + we
                    freq_list.append(comparison_results_final)
                    #we_vs_wd_labels = ['During the Weekday', 'on Saturdays', 'on Sundays']
                else:
                    pass
            else:
                pass

        # 2. Get work-life tweet frequency for weekdays in each lockdown date.
        freq_dic = {}
        for freq in range(len(freq_list)):
            df = freq_list[freq][0]
            if freq == len(freq_list)-1:
                freq_dic['freq_lockdown_{}'.format(freq+1)] = df[df['type'] == list(df['type'].unique())[2]] # for indexing final lockdown date
                freq_dic['freq_baseline_lockdown_{}'.format(freq+1)] = df[df['type'] == list(df['type'].unique())[0]] # for indexing baseline
                for k, v in freq_dic.items():
                    if 'baseline' in k:
                        freq_dic[k].loc[freq_dic[k].type == freq_dic[k].type.unique()[0], 'type'] = '(Baseline) Lockdown {}'.format(str(k)[-1:]) # add baseline label
                    else:
                        freq_dic[k].loc[freq_dic[k].type == freq_dic[k].type.unique()[0], 'type'] = 'Lockdown {}'.format(str(k)[-1:]) # add label
            else:
                freq_dic['freq_lockdown_{}'.format(freq+1)] = df[df['type'] == list(df['type'].unique())[1]] 
                freq_dic['freq_baseline_lockdown_{}'.format(freq+1)] = df[df['type'] == list(df['type'].unique())[0]] # for indexing baseline
                for k, v in freq_dic.items():
                    if 'baseline' in k:
                        freq_dic[k].loc[freq_dic[k].type == freq_dic[k].type.unique()[0], 'type'] = '(Baseline) Lockdown {}'.format(str(k)[-1:])
                    else:
                        freq_dic[k].loc[freq_dic[k].type == freq_dic[k].type.unique()[0], 'type'] = 'Lockdown {}'.format(str(k)[-1:])

        df = pd.concat(freq_dic.values())

        # 3. Create separate variables for lockdown and baseline dfs
        lockdowns = df[(df['type'] == 'Lockdown 1') | (df['type'] == 'Lockdown 2') | (df['type'] == 'Lockdown 3')]
        baselines = df[(df['type'] == '(Baseline) Lockdown 1') | (df['type'] == '(Baseline) Lockdown 2') | (df['type'] == '(Baseline) Lockdown 3')]

        # 4. Create a list of lockdown and baseline dfs
        df_dic = {'lockdowns': lockdowns, 'baselines': baselines}

        # 5. Groupby df separately for lockdowns and baselines. We do thisso we can plot lockdowns and baselines side-by-side
        for k, v in df_dic.items():
            norm_group = df_dic[k].groupby(['type'])['value'].sum().reset_index()
            norm_merge = pd.merge(df_dic[k], norm_group, on='type')
            norm_merge = norm_merge.rename(columns={'value_y': 'frequency_sum'})
            norm_merge['normalised_frequency'] =  norm_merge['value_x'] / norm_merge['frequency_sum']

            # Plot! 
            font_size = 16
            fig, ax = plt.subplots(len(norm_merge['type'].unique()), sharex=True, sharey=True, figsize=(8,8))
            plt.rcParams.update({'font.size': font_size})
            fig.text(0.04, 0.5, 'Normalised Frequency', va='center', rotation='vertical', fontsize=font_size)
            periods = list(norm_merge['type'].unique())
            mybins = np.arange(0,86400, 3600)
            for i in range(len(norm_merge['type'].unique())):
                _df = norm_merge[norm_merge['type'] == periods[i]]
                colour_dict = {'work': "C0", 'life': "C1"}
                for j in norm_merge['label'].unique():
                    bin_counts, bins = np.histogram(list(_df[_df['label'] == j]['value_x']), bins=mybins, density=True)
                    bin_centres = (bins[:-1] + bins[1:]) / 2
                    ax[i].errorbar(x=bin_centres, y=bin_counts, fmt='o-', capsize=2, alpha=1, label=j, color=colour_dict.get(j))
                    ax[i].set_xlim([0, 86400])
                    ax[i].xaxis.set_tick_params(labelsize=font_size)
                    ax[i].yaxis.set_tick_params(labelsize=font_size)
                    # increase size of scientific notation
                    ax[i].yaxis.get_offset_text().set_fontsize(font_size)
                xlim = np.arange(0, 60* 60 *24, 60*60*3)
                plt.ylim((0,0.000055))
                plt.xticks(xlim, [str(n).zfill(2) + ':00' for n in np.arange(0, 24, 3)])
                handles, labels = ax[i].get_legend_handles_labels()
                fig.legend(handles, labels, bbox_to_anchor=(1.04, 1))
                plt.xlabel('Time of Day', fontsize=font_size)
                plt.xticks(fontsize=font_size)
                plt.yticks(fontsize=font_size)
                ax[i].set_title('{}'.format(periods[i]), fontsize=font_size)
                plt.savefig('norm_frequency_comparison_plot_{}_{}_across_lockdowns.pdf'.format(community, k), format="pdf", bbox_inches='tight')
    
    def get_violinplot(self, single_user, day, thresholds, user, topic_labels, inc_topics, remove_legend, norm, flex):
        """ 
        Display a violin plot of work and life tweet activity for users. 

        Parameters
        ----------
            single_user (dictionary): A dictionary of labels/topics and their frequency values per day
            day (string): A string of a given day in the week (e.g., 'Monday')
            thresholds (list): A list of strings corresponding to beginning and end date of period
            user (int): An integer referring to the index of a specific user in 'result'. Used to index
                        'result' to obtain all tweets to plot for that user.
            periods1 (tuple): A tuple of information required for plotting - year, months, days, periods.
            labeldict2 (dict): A dictionary mapping BERTopic topic names (which are integers) to labels.
            inc_topics (bool): A boolean for specifying whether topic labels should be plotted.
                               If False, topics will be plotted without labels. If False in 
                               single_user_analysis() method, topics will not be plotted at all.

        Returns
        -------
            None
        """
        
        # Specify flag for including/excluding topic labels.
        if inc_topics:
            for i in list(topic_labels):
                single_user[topic_labels.get(i)] = single_user.pop(i)
        else:
            pass

        p_label = thresholds[0] + ' to ' + thresholds[1]
        keys = list(single_user.keys())

        arrays = list(single_user.values())
        max_length = 0
        for array in arrays:
            max_length = max(max_length, len(array))
        for array in arrays:
            array += [np.nan] * (max_length - len(array))
        
        dic = {}
        for k in keys:
            d = {k: {p_label: single_user.get(k)}}
            dic.update(d)

        df = pd.DataFrame(columns = ['label', p_label])
        for key, value in dic.items():
            period_df = pd.DataFrame({'label': [key]*len(value[p_label]),
                                    p_label: value[p_label]})
            df = df.append(period_df, ignore_index = True)
        
        df = pd.melt(frame = df, id_vars = 'label', value_vars = [p_label],
                    var_name = 'type', value_name = 'value')
        df['value'] = df['value'].astype(float)

        # Calculate Standard deviation for flexibility
        gr = df.groupby(['label', 'type'])['value'].mean().reset_index()
        flex_merge = pd.merge(df, gr, how='left', left_on=['label','type'], right_on=['label', 'type'])
        flex_concat = flex_merge.rename(columns={'value_x': 'value', 'value_y': 'mean'})
        flex_concat['domain_deviation'] = flex_concat['value'] - flex_concat['mean']
        flex_concat['domain_deviation_sq'] = flex_concat['domain_deviation']**2
        grouped_flex_sum = flex_concat.groupby(['label', 'type', 'mean'])['domain_deviation_sq'].sum().reset_index()
        grouped_flex_count = flex_concat.groupby(['label', 'type','mean'])['domain_deviation_sq'].count().reset_index()
        merged_grouped_flex = pd.merge(grouped_flex_sum, grouped_flex_count, how='left', left_on=['label','type'], right_on=['label', 'type'])
        merged_grouped_flex['var'] = merged_grouped_flex['domain_deviation_sq_x'] / merged_grouped_flex['domain_deviation_sq_y']
        merged_grouped_flex['flexibility_value'] = np.sqrt(merged_grouped_flex['var'])
        flex_df = pd.merge(merged_grouped_flex, flex_concat, how='left', left_on=['label','type'], right_on=['label', 'type'])
        flex_df = flex_df[['label', 'type', 'mean', 'flexibility_value']]

        flex_dict = dict(zip(flex_df['label'], flex_df['flexibility_value']))
        df['flexibility'] = df['label'].map(flex_dict)

        if norm:
            # Normalisation by day
            norm_group = df.groupby(['type'])['value'].sum().reset_index()
            norm_merge = pd.merge(df, norm_group, on='type')
            norm_merge = norm_merge.rename(columns={'value_y': 'frequency_sum'})
            norm_merge['normalised_frequency'] =  norm_merge['value_x'] / norm_merge['frequency_sum']
            work = norm_merge[norm_merge['label'] == 'work']
            life = norm_merge[norm_merge['label'] == 'life']

            work = list(work['value_x'])
            life = list(life['value_x'])
            mybins = np.arange(0,86400, 3600)

            fig, ax = plt.subplots()
            bin_counts, bins= np.histogram(work, bins=mybins, density=True)
            bin_centres = (bins[:-1] + bins[1:]) / 2
            y_error = np.random.rand(bin_counts.size)*10
            plt.errorbar(x=bin_centres, y=bin_counts, fmt='o-', capsize=2, label='work', color='fuchsia', alpha=1)
            bin_counts, bins= np.histogram(life, bins=mybins, density=True)
            bin_centres = (bins[:-1] + bins[1:]) / 2
            y_error = np.random.rand(bin_counts.size)*10
            plt.errorbar(x=bin_centres, y=bin_counts, fmt='o-', capsize=2, label='life', color='chartreuse', alpha=1)
            ax.set_xlim([0, 86400])
            xlim = np.arange(0, 60* 60 *24, 60*60*3)
            plt.ylim((0,0.000035))
            plt.xticks(xlim, [str(n).zfill(2) + ':00' for n in np.arange(0, 24, 3)])
            plt.xlabel('Time of Day')
            plt.ylabel('Normalised Frequency')
            plt.legend(title='Label')
            plt.title('Normalised Frequency Distribution of Work and Life Tweets for Users on \n {}s from {} to {}'.format(day, thresholds[0], thresholds[1]))
            plt.savefig('norm_ss_{}.pdf'.format(day), format="pdf")

        # Plotting
        fig, ax = plt.subplots()
        if flex:
            # plot flexibility line
            # xm = flex_df['mean'] + flex_df['flexibility_value']
            # ax.hlines(y='label', xmin='mean', xmax=xm, data=df, color='black', linewidth=1)
            _min = flex_df['mean'] - flex_df['flexibility_value']
            _max = flex_df['mean'] + flex_df['flexibility_value']
            ax.hlines(y='label', xmin=_min, xmax=_max, data=flex_df, color='black', linewidth=1)
        sns.pointplot(x = 'value', y='label', data=df, estimator=mean, join=False, 
        ci=None, markers='x', color='red', linestyles='', capsize=0.3, zorder=7)
        plt.setp(ax.lines, zorder=100)
        plt.setp(ax.collections, zorder=100, label="")
        sns.violinplot(ax=ax, data = df, x='value', y='label', 
        split = False, cut=0, inner=None, linewidth=0, saturation=0.4, color="0.4", palette="Set2")
        
        # sns.boxplot(ax=ax, x='value', y= "label", data=df, width=0.3,
        #     boxprops={'zorder': 2}, color="0.2", palette="Set2")
        # Add 'jitter' to violinplots
        sns.stripplot(x="value", y="label", data=df, jitter=True)
        
        try:
            if remove_legend:
                ax.get_legend().remove()
            else:
                pass
        except:
            pass

        ax.set_xlim([0, 86400])
        xlim = np.arange(0, 60* 60 *24, 60*60*3)
        plt.xticks(xlim, [str(n).zfill(2) + ':00' for n in np.arange(0, 24, 3)])
        plt.xlabel('Time of Day')
        plt.ylabel('Label')
        plt.title('Distribution of Work and Life Tweets for Users on \n {}s from {} to {}'.format(day, thresholds[0], thresholds[1]))
        plt.tight_layout()
        if flex:
            plt.savefig('outputs/wlb_plots/flex_frequency_plot_{}.pdf'.format(day.lower()), format="pdf")
        else:
            plt.savefig('outputs/wlb_plots/frequency_plot_{}.pdf'.format(day.lower()), format="pdf")

    def run_statistical_test(self, comparison_results_final, wd_vs_we):
        """ 
        Statistical validation of results:
        Compute a Wilcoxon rank-sum test and Kolmogorov-Smirnov test to test the null hypothesis that two
        related paired samples (vectors) come from the same distribution

        Parameters
        ----------
            comparison_results (dictionary): A nested dictionary of work-life frequency results {period: {day: {label: [vals]}}}
            wd_vs_we (bool): If True, concatenate work-life tweet frequencies for all weekdays.

        Returns
        -------
            mann_whtiney (dictionary): A dictionary storing results from mann_whitney test
            ks (dictionary): A dictionary storing results from kolmogorov-smirnov test 
        """

        # NB: Specify a condition for plotting weekdays vs weekends.
        # This is additional to plotting daily results
        if wd_vs_we:
            if len(comparison_results_final) == 7:
                wd = [pd.concat(comparison_results_final[0:4])]
                we = comparison_results_final[5:7] 
                comparison_results_final = wd + we
                #we_vs_wd_labels = ['During the Weekday', 'on Saturdays', 'on Sundays']
            else:
                pass
        else:
            pass

        # 1. Create dictionary to store work and life frequency results
        work_vector = defaultdict(list)
        life_vector = defaultdict(list)
        for day in range(len(comparison_results_final)):
            period_list = list(comparison_results_final[day]['type'].unique())
            for period in range(len(period_list)):
                sub_df = comparison_results_final[day][comparison_results_final[day]['type'] == period_list[period]]
                sub_df = sub_df[sub_df['value'].notna()]
                work = sub_df[sub_df['label'] == 'work']
                life = sub_df[sub_df['label'] == 'life']
                work_vector[day].append(list(work['value']))
                life_vector[day].append(list(life['value']))
        dic = {'work': dict(work_vector), 'life': dict(life_vector)}

        # 2. Compute Wilcoxon-Mann-Whitney and Kolmogorov-Smirnov Test for
        #  work frequency vectors and life frequency vectors for each day 
        # across periods.
        mann_whitney = defaultdict(lambda: defaultdict(list))
        ks = defaultdict(lambda: defaultdict(list))
        for k, v in dic.items():
            for day, vals in dic[k].items():
                for i, j in zip(dic[k][day][0::1], dic[k][day][1::1]):
                    mann_whitney[k][day].append(stats.mannwhitneyu(i, j)[1]) # test for location, e.g. differences in mean/median
                    ks[k][day].append(stats.ks_2samp(i, j)[1]) # test for scale, e.g. diffferences in IQR
        mann_whitney = {
            label: {
                day: test_results for day, test_results in label_dic.items()
            } for label, label_dic in mann_whitney.items()
        }
        ks = {
            label: {
                day: test_results for day, test_results in label_dic.items()
            } for label, label_dic in ks.items()
        }

        return mann_whitney, ks

    def compute_perm_flex(self, q1, q3, comparison_results_final, wd_vs_we):
        """ 
        Compute descriptive statistics representing flexibility and permeability
        of borders for work and life tweets.

        Parameters
        ----------
            comparison_results (dictionary): A nested dictionary of work-life frequency results {period: {day: {label: [vals]}}}

        Returns
        -------
            perm (str): A string of dataframes containing permeability results
            flex (str): A string of dataframes containing flexibility results
        """

        # NB: Specify a condition for plotting weekdays vs weekends.
        # This is additional to plotting daily results
        if wd_vs_we:
            if len(comparison_results_final) == 7:
                wd = [pd.concat(comparison_results_final[0:4])]
                we = comparison_results_final[5:7] 
                comparison_results_final = wd + we
                #we_vs_wd_labels = ['During the Weekday', 'on Saturdays', 'on Sundays']
            else:
                pass
        else:
            pass
        
        flex = []
        perm = []
        for df in range(len(comparison_results_final)):
            g1 = comparison_results_final[df].groupby(['type', 'label']).agg({'value': ['mean', 'std', 'median', 'count' , 'max', 'min', q1, q3]})
            m1 = pd.merge(comparison_results_final[df], g1, on=['type', 'label'])
            m1 = m1.rename(columns={('value', 'mean'): 'mean',('value', 'std'): 'standard_deviation', 
            ('value', 'median'): 'median', ('value', 'q1'): 'Q1', ('value', 'q3'): 'Q3', 
            ('value', 'count'): 'count', ('value', 'max'): 'max', ('value', 'min'): 'min'})
            m1['IQR'] = m1['Q3'] - m1['Q1']
            m1['Q0'] = m1['min']
            m1['Q4'] = m1['max']
            m1['whisker'] = m1['Q4'] - m1['Q0']
            m1['2hrs_data'] = m1['value'].between(m1['mean']-(3600*2), (3600*2)+m1['mean'])
            m1['4hrs_data'] = m1['value'].between(m1['mean']-(3600*4), (3600*4)+m1['mean'])
            m1['6hrs_data'] = m1['value'].between(m1['mean']-(3600*6), (3600*6)+m1['mean'])
            m1['8hrs_data'] = m1['value'].between(m1['mean']-(3600*8), (3600*8)+m1['mean'])
            m1['2hrs'] = 3600*2
            m1['4hrs'] = 3600*4
            m1['6hrs'] = 3600*6
            m1['8hrs'] = 3600*8
            m1 = m1.sort_values('label')
            m1['median_time'] = pd.to_datetime(m1['median'], unit='s')
            m1['Q0_time'] = pd.to_datetime(m1['Q0'], unit='s')
            m1['Q1_time'] = pd.to_datetime(m1['Q1'], unit='s')
            m1['Q3_time'] = pd.to_datetime(m1['Q3'], unit='s')
            m1['Q4_time'] = pd.to_datetime(m1['Q4'], unit='s')
            m1['IQR_time'] = pd.to_datetime(m1['IQR'], unit='s')
            m1['whisker_time'] = pd.to_datetime(m1['whisker'], unit='s')
            final = m1.groupby(['type', 'label', 'mean', 'standard_deviation', 
            'median', 'median_time', '2hrs', '4hrs', '6hrs', '8hrs', 'count', 
            'Q0_time', 'Q1_time', 'Q1', 'Q3_time', 'Q3', 'Q4_time', 'IQR_time',
             'whisker_time'])['2hrs_data', '4hrs_data', '6hrs_data', '8hrs_data'].sum().reset_index()
            final['2hrs_percentage'] = final['2hrs_data'] / final['count']
            final['4hrs_percentage'] = final['4hrs_data'] / final['count']
            final['6hrs_percentage'] = final['6hrs_data'] / final['count']
            final['8hrs_percentage'] = final['8hrs_data'] / final['count']
            final['year'] = final['type'].apply(lambda x: x[-4:])
            final = final.sort_values(['year'], ascending=True)

            # Permeability Calculation
            # p = final.groupby(final.index//2)['median_time'].diff(-1).dropna().reset_index(drop=True).abs()
            p = final.groupby(['type']).agg({'Q1': ['min', 'max'], 'Q3': ['min', 'max']})
            p.columns = ["_".join(x) for x in p.columns.ravel()]
            p['overlap'] = p['Q3_min'] - p['Q1_max']
            p['joint_iqr'] = p['Q3_max'] - p['Q1_min']
            p['permeability (%)'] = (p['overlap'] / p['joint_iqr']) * 100
            p = p['permeability (%)']
            final = pd.merge(final, p, on=['type'])
            # str(timedelta(seconds=(min_of_Q3 - max_of_Q1)))

            # Flexibility Calculation
            f = final[['type', 'label', 'count', 'median_time', 'Q1_time', 'Q3_time', 'IQR_time']]

            perm.append(final)
            flex.append(f)

        return perm, flex

    def compute_swb(self, comparison_results_final, wd_vs_we, work_vs_life):
        """ 
        Computes the subjective well-being (swb) value for the entire community of 
        Twitter users. We compute the swb daily for a a given period.
        If work_vs_life == True, computes subjective well-being separately for work and life.

        Parameters
        ----------
            comparison_results (list): A list of dataframes comprising label, type, tweet_frequency and tweets for
                                       each day in a given period.
            wd_vs_we (bool): Whether you want to compare weekdays vs weekend (True/False)
            work_vs_life (bool): Whether you want to compute subjective well-being separately for work and life
        Returns
        -------
            sentiment_df (pandas df): The subjective well-being score for each day for each period.
            work_vs_life_dic (dict): The subjective well-being score for each day for each period separated by work and life.
        """
        
        # NB: Specify a condition for plotting weekdays vs weekends.
        # This is additional to plotting daily results
        if wd_vs_we:
            if len(comparison_results_final) == 7:
                wd = [pd.concat(comparison_results_final[0:4])]
                we = comparison_results_final[5:7] 
                comparison_results_final = wd + we
                we_vs_wd_labels = ['During the Weekday', 'on Saturdays', 'on Sundays']
            else:
                pass
        else:
            pass

        sid = SentimentIntensityAnalyzer()
        work_vs_life_dic = defaultdict(list)
        sentiment_df = []
        for df in comparison_results_final:
            if work_vs_life:
                for i in range(len(list(df['label'].unique()))):
                    dff = df[df['label'] == list(df['label'].unique())[i]]
                    dff['tweet'] = dff['tweet'].astype(str)
                    dff['sentiment_scores'] = dff['tweet'].apply(lambda tweet: sid.polarity_scores(tweet))
                    dff['compound'] = dff['sentiment_scores'].apply(lambda score_dict: score_dict['compound'])
                    dff['comp_score'] = dff['compound'].apply(lambda c: 'pos' if c >=0 else 'neg')
                    dff['index_count'] = range(len(dff.index))
                    dff = dff.dropna()
                    sentiment_count_groups = (dff.groupby(['type', 'comp_score', 'author_id']).agg(sentiment_count=('index_count', 'nunique')).reset_index())
                    sentiment_count_groups['year'] = sentiment_count_groups['type'].apply(lambda x: x[-4:])
                    sentiment_count_groups = sentiment_count_groups.sort_values(['year'], ascending=True)
                    
                    # Compute swb and return it
                    sentiment_count_groups['swb_numerator'] = sentiment_count_groups['sentiment_count'] - sentiment_count_groups['sentiment_count'].shift()
                    sentiment_count_groups['swb_denominator'] = sentiment_count_groups['sentiment_count'] + sentiment_count_groups['sentiment_count'].shift()
                    sentiment_count_groups['swb'] = sentiment_count_groups['swb_numerator'] / sentiment_count_groups['swb_denominator']
                    swb = sentiment_count_groups[1::2]
                    swb = swb[['type', 'author_id', 'swb']]
                    work_vs_life_dic[list(df['label'].unique())[i]].append(swb)
                return dict(work_vs_life_dic)
            else:
                df['tweet'] = df['tweet'].astype(str)
                df['sentiment_scores'] = df['tweet'].apply(lambda tweet: sid.polarity_scores(tweet))
                df['compound']  = df['sentiment_scores'].apply(lambda score_dict: score_dict['compound'])
                df['comp_score'] = df['compound'].apply(lambda c: 'pos' if c >=0 else 'neg')
                df['index_count'] = range(len(df.index))
                # NB: We have nan values because, during the calculation of the daily tweet frequency,
                # we need to ensure all values for each period are the same length. We can drop these here because
                # we are only interested in the tweets, not their frequency.
                df = df.dropna()
                sentiment_count_groups = (df.groupby(['type', 'comp_score', 'author_id']).agg(sentiment_count=('index_count', 'nunique')).reset_index())
                sentiment_count_groups['year'] = sentiment_count_groups['type'].apply(lambda x: x[-4:])
                sentiment_count_groups = sentiment_count_groups.sort_values(['year'], ascending=True)

                # Compute swb and return it
                sentiment_count_groups['swb_numerator'] = sentiment_count_groups['sentiment_count'] - sentiment_count_groups['sentiment_count'].shift()
                sentiment_count_groups['swb_denominator'] = sentiment_count_groups['sentiment_count'] + sentiment_count_groups['sentiment_count'].shift()
                sentiment_count_groups['swb'] = sentiment_count_groups['swb_numerator'] / sentiment_count_groups['swb_denominator']
                swb = sentiment_count_groups[1::2]
                swb = swb[['type', 'author_id', 'swb']]
                sentiment_df.append(swb)
                return sentiment_df

    def get_work_life_results(self, result, year, month, week, date_threshold, topic_labels, all_users=True, include_topics=False, single_period=False):
        """ 
        Computes work-life analysis. Produces plots for the border theory operationalization paper.

        Parameters
        ----------
            ...
        -------
            comparison_results_final (pandas.core.frame.DataFrame): A dataframe with len=7 providing work-life
                classification of tweets for each day in the week.
        """

        # 3. Specify whether you want to analyse all users or not
        if all_users:
            user = 0
            df = pd.concat(result)
        else:
            user = 0
            df = result[user]

        # 4. Get work-life tweet frequency for each day througout a given period
        weekly_results = defaultdict(list)
        for day in week:
            for period in range(len(year)):
                daily_tweet_volume = self.compute_work_life_frequency(user, df, year[period], 
                month[period], day, date_threshold[period], inc_topics=include_topics, _continuum=False)
                weekly_results[period].append(daily_tweet_volume)
        weekly_results = dict(weekly_results)

        # 5. Generate violin plots displaying results for a single period 
        # or comparing results across N periods.
        if single_period:
            # Get violinplot for each day in a week. 
            for period in range(len(weekly_results)):
                for day in range(len(weekly_results.get(period))):
                    violin = self.get_violinplot(weekly_results.get(period)[day], week[day], 
                    date_threshold[period], user, topic_labels, inc_topics=include_topics, 
                    remove_legend=True, norm=False, flex=False)
        else:
            # i. Compare work-life tweet frequency across multiple periods
            comparison_results, periods = self.compare_work_life_frequency(weekly_results, week=week,
            thresholds=date_threshold, inc_topics=include_topics, norm=False, flex=False, 
            topic_labels=topic_labels, wd_vs_we=True)

        # # Update (30/12/22)
        # # ii. comparison_results now contains tweet frequency with corresponding tweet text.
        # #     We must split tweet frequency from tweet data for our plotting code to work.
        comparison_results_final = []
        for df in comparison_results:
            labels = pd.concat([df[df['label'] == 'work'], df[df['label'] == 'life']], ignore_index=True)
            labels.reset_index(inplace = True)
            tweets = pd.concat([df[df['label'] == 'work_tweet'], df[df['label'] == 'life_tweet']], ignore_index=True)
            tweets.reset_index(inplace = True)
            authors = pd.concat([df[df['label'] == 'work_author_id'], df[df['label'] == 'life_author_id']], ignore_index=True)
            authors.reset_index(inplace = True)
            df_update = labels.merge(tweets, on='index').merge(authors, on='index')
            df_update = df_update.rename(columns={'label': 'label_id', 'label_x': 'label', 'value_x': 'value', 'value_y': 'tweet', 'value': 'author_id'})
            df_update = df_update.drop(columns=['index', 'type_x', 'label_id', 'label_y', 'type_y'])
            comparison_results_final.append(df_update)

        return comparison_results_final

    def plot_perm_flex_swb(self, comparison_results_final_list, swb_list, community, work_vs_life):
        """ 
        Produces plots for IQR of tweet frequency and subjective well-being. 
        NB: These are the main plots of the border theory operationalization paper.

        Parameters
        ----------
            comparison_results_final_list (nested list): A nested list of tweet frequencies,
                classified as either work or life, for each day of the week, for each lockdown.
            swb_list (list): A list of the subjective well-being values for each lockdown
            community (string): The community we are conducting the analysis for
            work_vs_life (bool): If True, we analyse the subjective well-being separately for work and life tweets.
        -------
            comparison_results_final (pandas.core.frame.DataFrame): A dataframe with len=7 providing work-life
                classification of tweets for each day in the week.
        """

        # 1. Ensure You concatenate all weekdays into one dataframe for comparison_results_final.
        freq_list = []
        for lockdown_result in comparison_results_final_list:
            comparison_results_final = lockdown_result
            wd_vs_we=True
            if wd_vs_we:
                if len(comparison_results_final) == 7:
                    wd = [pd.concat(comparison_results_final[0:4])]
                    we = comparison_results_final[5:7] 
                    comparison_results_final = wd + we
                    freq_list.append(comparison_results_final)
                    #we_vs_wd_labels = ['During the Weekday', 'on Saturdays', 'on Sundays']
                else:
                    pass
            else:
                pass
        
        # 2. Get work-life tweet frequency for weekdays in each lockdown date.
        # NB: Remember to select ['type'].unique()[2] for final lockdown, which is from 06 Jan 2021 to 08 March 2021
        freq_dic = {}
        for freq in range(len(freq_list)):
            df = freq_list[freq][0]
            if freq == len(freq_list)-1:
                freq_dic['freq_lockdown_{}'.format(freq+1)] = df[df['type'] == list(df['type'].unique())[2]] # for indexing final lockdown date
                freq_dic['freq_baseline_lockdown_{}'.format(freq+1)] = df[df['type'] == list(df['type'].unique())[0]] # for indexing baseline
                for k, v in freq_dic.items():
                    if 'baseline' in k:
                        freq_dic[k].loc[freq_dic[k].type == freq_dic[k].type.unique()[0], 'type'] = ' (Baseline) \n Lockdown {}'.format(str(k)[-1:]) # add baseline label
                    else:
                        freq_dic[k].loc[freq_dic[k].type == freq_dic[k].type.unique()[0], 'type'] = 'Lockdown {}'.format(str(k)[-1:]) # add label
            else:
                freq_dic['freq_lockdown_{}'.format(freq+1)] = df[df['type'] == list(df['type'].unique())[1]] 
                freq_dic['freq_baseline_lockdown_{}'.format(freq+1)] = df[df['type'] == list(df['type'].unique())[0]] # for indexing baseline
                for k, v in freq_dic.items():
                    if 'baseline' in k:
                        freq_dic[k].loc[freq_dic[k].type == freq_dic[k].type.unique()[0], 'type'] = ' (Baseline) \n Lockdown {}'.format(str(k)[-1:])
                    else:
                        freq_dic[k].loc[freq_dic[k].type == freq_dic[k].type.unique()[0], 'type'] = 'Lockdown {}'.format(str(k)[-1:])
                
        # 3. Get swb for weekdays in each lockdown date.
        # If work_vs_life=True, we get the swb separately for work and life tweets
        # NB: Remember to select ['type'].unique()[2] for final lockdown, which is from 06 Jan 2021 to 08 March 2021
        if work_vs_life:
            swb_dic = defaultdict(lambda: defaultdict(list))
            for swb in range(len(swb_list)):
                label_dic = swb_list[swb]
                for label, swb_values in label_dic.items():
                    df = swb_list[swb][label][0]
                    if swb == len(swb_list)-1: # we get the swb separately for the final lockdown
                        swb_dic[label]['swb_lockdown_{}'.format(swb+1)].append(df[df['type'] == list(df['type'].unique())[2]]) # for indexing final lockdown date
                        swb_dic[label]['swb_baseline_lockdown_{}'.format(swb+1)].append(df[df['type'] == list(df['type'].unique())[0]]) # for indexing baseline
                        for label, label_dic in swb_dic.items():
                            for k, v in label_dic.items():
                                if 'baseline' in k:
                                    swb_dic[label][k][0].loc[swb_dic[label][k][0].type == swb_dic[label][k][0].type.unique()[0], 'type'] = ' (Baseline) \n Lockdown {}'.format(str(k)[-1:])
                                else:
                                    swb_dic[label][k][0].loc[swb_dic[label][k][0].type == swb_dic[label][k][0].type.unique()[0], 'type'] = 'Lockdown {}'.format(str(k)[-1:]) # rename cols
                    else:
                        swb_dic[label]['swb_lockdown_{}'.format(swb+1)].append(df[df['type'] == list(df['type'].unique())[2]]) # for indexing final lockdown date
                        swb_dic[label]['swb_baseline_lockdown_{}'.format(swb+1)].append(df[df['type'] == list(df['type'].unique())[0]]) # for indexing baseline
                        for label, label_dic in swb_dic.items():
                            for k, v in label_dic.items():
                                if 'baseline' in k:
                                    swb_dic[label][k][0].loc[swb_dic[label][k][0].type == swb_dic[label][k][0].type.unique()[0], 'type'] = ' (Baseline) \n Lockdown {}'.format(str(k)[-1:])
                                else:
                                    swb_dic[label][k][0].loc[swb_dic[label][k][0].type == swb_dic[label][k][0].type.unique()[0], 'type'] = 'Lockdown {}'.format(str(k)[-1:]) # rename cols
            swb_dic = {label: {lockdown: swb_vals[0] for lockdown, swb_vals in label_dic.items()} for label, label_dic in swb_dic.items()}
        else:
            swb_dic = {}
            for swb in range(len(swb_list)):
                df = swb_list[swb][0]
                if swb == len(swb_list)-1:
                    swb_dic['swb_lockdown_{}'.format(swb+1)] = df[df['type'] == list(df['type'].unique())[2]] # for indexing final lockdown date
                    swb_dic['swb_baseline_lockdown_{}'.format(swb+1)] = df[df['type'] == list(df['type'].unique())[0]] # for indexing baseline
                    for k, v in swb_dic.items():
                        if 'baseline' in k:
                            swb_dic[k].loc[swb_dic[k].type == swb_dic[k].type.unique()[0], 'type'] = ' (Baseline) \n Lockdown {}'.format(str(k)[-1:])
                        else:
                            swb_dic[k].loc[swb_dic[k].type == swb_dic[k].type.unique()[0], 'type'] = 'Lockdown {}'.format(str(k)[-1:]) # rename cols
                else:
                    swb_dic['swb_lockdown_{}'.format(swb+1)] = df[df['type'] == list(df['type'].unique())[1]] # for indexing all other lockdown dates
                    swb_dic['swb_baseline_lockdown_{}'.format(swb+1)] = df[df['type'] == list(df['type'].unique())[0]] # for indexing baseline
                    for k, v in swb_dic.items():
                        if 'baseline' in k:
                            swb_dic[k].loc[swb_dic[k].type == swb_dic[k].type.unique()[0], 'type'] = ' (Baseline) \n Lockdown {}'.format(str(k)[-1:])
                        else:
                            swb_dic[k].loc[swb_dic[k].type == swb_dic[k].type.unique()[0], 'type'] = 'Lockdown {}'.format(str(k)[-1:]) # rename cols

        # 4. Concatenate tweet frequency and swb datasets (freq_dic.values() and swb_dic.values())
        comparison_results_all = pd.concat(freq_dic.values())
        if work_vs_life:
            swb_all = {}
            for label, label_dic in swb_dic.items():
                swb_all[label] = pd.concat(swb_dic[label].values())
        else:
            swb_all = pd.concat(swb_dic.values())

        # 5. Plot!
        sns.set_palette("pastel")
        colour='black'
        df = comparison_results_all

        # create subplots
        fig, axs = plt.subplots(nrows=2, figsize=(16, 12)) # figsize=(9,12)
        sns.set_context("paper")
        df = df.rename(columns={"label": "Category"})

        # NB: We plot the IQR frequency and subjective well-being plots together as subplots.
        # 4.1 Plot the IQR frequencies (subplot 1)
        sns.boxplot(y='value', x='type', data=df, hue='Category', whis=0, showfliers=False, ax=axs[0])
        #sns.move_legend(axs[0], "upper left", bbox_to_anchor=(1, 1))
        axs[0].set_ylim([86400, 0])
        ylim = np.arange(0, 60* 60 *24, 60*60*3)
        axs[0].set_yticks(ylim)
        axs[0].set_yticklabels([str(n).zfill(2) + ':00' for n in np.arange(0, 24, 3)])
        axs[0].set_ylabel('Time of Day')
        axs[0].set_xlabel('Period')

        # 4.2 Plot the subjective well-being (subplot 2)
        if work_vs_life:
            for label, label_dic in swb_all.items():
                # create subplots
                fig, axs = plt.subplots(nrows=2, figsize=(16, 12)) # figsize=(9,12)
                sns.set_context("paper")
                df = df.rename(columns={"label": "Category"})
                # NB: We plot the IQR frequency and subjective well-being plots together as subplots.
                # 4.1 Plot the IQR frequencies (subplot 1)
                sns.boxplot(y='value', x='type', data=df, hue='Category', whis=0, showfliers=False, ax=axs[0])
                #sns.move_legend(axs[0], "upper left", bbox_to_anchor=(1, 1))
                axs[0].set_ylim([86400, 0])
                ylim = np.arange(0, 60* 60 *24, 60*60*3)
                axs[0].set_yticks(ylim)
                axs[0].set_yticklabels([str(n).zfill(2) + ':00' for n in np.arange(0, 24, 3)])
                axs[0].set_ylabel('Time of Day')
                axs[0].set_xlabel('Period')

                sns.boxplot(y='swb', x='type', data=swb_all[label], saturation=0.6, ax=axs[1]) # if error saying '1', DON'T index sentiment_df
                axs[1].set_xlabel('Period')
                axs[1].set_ylabel('Subjective Well-Being Score')
                for i, box in enumerate(axs[1].artists):
                    box.set_edgecolor('black')
                    box.set_facecolor('white')
                    # iterate over whiskers and median lines
                    for j in range(6*i,6*(i+1)):
                        axs[1].lines[j].set_color('black')
                self.wrap_x_labels(axs[1], 12) 
                axs[1].figure

                fig.savefig(
                "{}_{}_swb_all_lockdowns.pdf".format(community, label),
                # we need a bounding box in inches
                bbox_inches=mtransforms.Bbox(
                    # This is in "figure fraction" for the bottom half
                    # input in [[xmin, ymin], [xmax, ymax]]
                    [[0.06, 0.05], [0.91, 0.47]]
                ).transformed(
                    # this take data from figure fraction -> inches
                    #    transFigrue goes from figure fraction -> pixels
                    #    dpi_scale_trans goes from inches -> pixels
                    (fig.transFigure - fig.dpi_scale_trans)
                ),
            )
                fig.savefig(
                "{}_flex_all_lockdowns.pdf".format(community),
                bbox_inches=mtransforms.Bbox([[0.06, 0.47], [0.91, 0.9]]).transformed(
                    fig.transFigure - fig.dpi_scale_trans
                ),
            )
        else:
            sns.boxplot(y='swb', x='type', data=swb_all, saturation=0.6, ax=axs[1]) # if error saying '1', DON'T index sentiment_df
            axs[1].set_xlabel('Period')
            axs[1].set_ylabel('Subjective Well-Being Score')
            for i, box in enumerate(axs[1].artists):
                box.set_edgecolor('black')
                box.set_facecolor('white')
                # iterate over whiskers and median lines
                for j in range(6*i,6*(i+1)):
                    axs[1].lines[j].set_color('black')
            self.wrap_x_labels(axs[1], 12) 
            axs[1].figure

        def get_axis_limits(ax, scale=.9):
            return ax.get_xlim()[1]*scale, ax.get_ylim()[1]*scale

        fig.savefig(
            "{}_swb_all_lockdowns.pdf".format(community),
            # we need a bounding box in inches
            bbox_inches=mtransforms.Bbox(
                # This is in "figure fraction" for the bottom half
                # input in [[xmin, ymin], [xmax, ymax]]
                [[0.06, 0.05], [0.91, 0.47]]
            ).transformed(
                # this take data from figure fraction -> inches
                #    transFigrue goes from figure fraction -> pixels
                #    dpi_scale_trans goes from inches -> pixels
                (fig.transFigure - fig.dpi_scale_trans)
            ),
        )

        fig.savefig(
            "{}_flex_all_lockdowns.pdf".format(community),
            bbox_inches=mtransforms.Bbox([[0.06, 0.47], [0.91, 0.9]]).transformed(
                fig.transFigure - fig.dpi_scale_trans
            ),
        )

    def get_crisis_information(self, l1=False, l2=False, l3=False, all_lockdown_info=False):
        """ 
        Returns key variables for specific lockdown period. Variables
        are required to run method that computes work-life analysis results.
        # Note the following important periods:
            # National Lockdown 1: 2020-03-23 to 2020-06-01 (phased re-opening of schools)
            # National Lockdown 2: 2020-11-05 to 2020-12-02 (return to three-tier system of restrictions, but no longer in national lockdown)
            # National Lockdown 3: 2021-01-06 to 2021-03-08 (re-opening of primary and secondary schools)
            # NB: You compare results for UK national lockdowns. Lockdown dates may be slightly different in Scotland
            # and Ireland, but we want to compare results for Scottish and Irish teachers using the same periods.

        Parameters
        ----------
            l1 (bool): Set True if you want information for lockdown 1. False, otherwise
            l2 (bool): Set True if you want information for lockdown 2. False, otherwise
            l3 (bool): Set True if you want information for lockdown 3. False, otherwise
        -------
            comparison_results_final (pandas.core.frame.DataFrame): A dataframe with len=7 providing work-life
                classification of tweets for each day in the week.
        """
        
        # 1. Define a list of lists corresponding to each year under study.
        y = [[2019], [2020], [2021], [2022]]

        # 2. Define a list of lists corresponding to months for each lockdown period.
        l1_months = list([['March', 'April', 'May']]*4)
        l2_months = list([['November', 'December']]*4)
        l3_months = list([['January', 'February', 'March']]*4)

        # 3. Define a list of lists corresponding to specific dates of each lockdown period.
        l1_thresh = list(['{}-03-23'.format(year), '{}-06-01'.format(year)] for year in [x[0] for x in y])
        l2_thresh = list(['{}-11-05'.format(year), '{}-12-02'.format(year)] for year in [x[0] for x in y])
        l3_thresh = list(['{}-01-06'.format(year), '{}-03-08'.format(year)] for year in [x[0] for x in y])

        # 4. Define a list corresponding to each day in a week.
        w = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

        if l1:
            return y, l1_months, w, l1_thresh
        elif l2:
            return y, l2_months, w, l2_thresh
        elif l3:
            return y, l3_months, w, l3_thresh,
        elif all_lockdown_info:
            return [[y, l1_months, w, l1_thresh], [y, l2_months, w, l2_thresh], [y, l3_months, w, l3_thresh]]
        
    def wrap_x_labels(self, ax, width, break_long_words=False):
        labels = []
        for label in ax.get_xticklabels():
            text = label.get_text()
            labels.append(textwrap.fill(text, width=width,
                        break_long_words=break_long_words))
        ax.set_xticklabels(labels, rotation=0)

# NB: Additional functions to enable the analysis
def q1(x):
    # return bottom 25% quantile of some data, x
    return x.quantile(0.25)

def q3(x):
     # return top 75% quantile of some data, x
    return x.quantile(0.75)

def read_pickle_dic(name): 
    """
    Read a pickled dictionary/list/etc. file on
    a local machine

    Parameters
    ----------
        name (str): The name of the .pkl file

    Returns
    -------
        dic_in (dic): The loaded dictionary 
    """

    pickle_in = open("{}.pkl".format(name), "rb")
    dic_in = pickle.load(pickle_in)
    pickle_in.close()
    
    return dic_in

"""
Instantiate topic_analysis class below
"""

# 0. Initialise instance variables
# NB: Choose which community to run the analysis on
community = 'Journalists' # ScottishTeachers # IrishTeachers # Journalists
_dir = ''
bertmodel = BERTopic.load(_dir + 'BERTopic/{}/{}_nouns_model'.format(community, community))
tweets = read_pickle_dic(_dir + 'BERTopic/{}/{}_tweets'.format(community, community))
topics = read_pickle_dic(_dir + 'BERTopic/{}/{}_topics'.format(community, community))
timestamps = read_pickle_dic(_dir + 'BERTopic/{}/{}_timestamps'.format(community, community))
embeddings = read_pickle_dic(_dir + 'BERTopic/{}/{}_embeddings'.format(community, community))

# 1. Instantiate topic_analysis class
wl = work_life_analysis(_dir, community, bertmodel, tweets, topics, timestamps, embeddings)

# OPTIONAL: Quantitative Topic Evaluation
topic_diversity, topic_significance = wl.evaluate_model(bertmodel)

# 2. Map topics back to original data
orig_merge = wl.map_topics()

# 3. Define topic labels. After manually inspecting the BERTopic topics, you must group topics into work and life by 
# modifying the relevant dictionary below.
if community == 'ScottishTeachers':
    labels = {
        'work': 
        {0: 'Teaching', 3: 'Education'}, 
        'life': 
        {1: 'Entertainment', 2: 'Politics', 4: 'Days of the Week'}
    }
elif community == 'IrishTeachers':
    labels = {
    'work': 
    {0: 'Education', 4: 'Curriculum'}, 
    'life': 
    {1: 'Teaching Unions', 2: 'Politics', 3: 'Life'}
}
elif community == 'Journalists':
    labels = {
    'work': 
    {0: 'COVID-19 Coverage', 1: 'General News Coverage', 4: 'Sports Coverage'}, 
    'life': 
    {2: 'Entertainment', 3: 'Life'}
}

label_dict = {}
for k, v in labels.items():
    label_dict.update(v)

topic_labels = {}
for k, v in label_dict.items():
    topic_labels.update({str(k):v})

# 3. Transform data (separate 'work' from 'life' tweets)
result = wl.transform_data(orig_merge, list(labels.get('work').keys()), 
list(labels.get('life').keys())) # orig_merge # utc_offset

"""
*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
WORK-LIFE TWEET ACTIVITY ANALYSIS 
*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

Here, we analyse the work-life tweet activity for either a single user
or all users. 
    - Single user: This can be used to check analysis for specific use cases
    - All users: This is the main analysis. Barycenter used to compare work and life
For example, below we analyse the work-life tweet activity for all users 
across each of the three national UK lockdown periods.
"""

# 1. Iterate through all lockdowns to get border and subjective well-being informaiton for all lockdowns.
all_lockdowns = wl.get_crisis_information(all_lockdown_info=True)

# 2. Get work-life tweet frequency and subjective well-being results for all lockdown periods. Here, we iterate through all lockdowns.
comparison_results_final_list = []
swb_list = []
border_permeability = []
for lockdown in range(len(all_lockdowns)):
    comparison_results_final = wl.get_work_life_results(result, all_lockdowns[lockdown][0],all_lockdowns[lockdown][1], 
                                                        all_lockdowns[lockdown][2], all_lockdowns[lockdown][3], topic_labels)
    swb_list.append(wl.compute_swb(comparison_results_final, wd_vs_we=True, work_vs_life=False))
    comparison_results_final_list.append(comparison_results_final)
    perm, flex = wl.compute_perm_flex(q1, q3, comparison_results_final, wd_vs_we=True) # comparison_results
    border_permeability.append(perm)

# 3. Plot tweet frequency (IQR boxplots) and subjective well-being (boxplots) for each lockdown period.
wl.plot_perm_flex_swb(comparison_results_final_list, swb_list, community, work_vs_life=False)

# 4. Plot the normalised work-life tweet frequency for all lockdown periods.
wl.plot_work_norm_frequency(comparison_results_final_list, community)

# 5. Compute permeability and flexibility of tweet frequency
# NB: This is used for Tables 9, 10, and 11 in the paper (permeability values per period).
perm, flex = wl.compute_perm_flex(q1, q3, comparison_results_final, wd_vs_we=True) # comparison_results

# OPTIONAL
# Compute statistical tests
# mann_whitney, ks = wl.run_statistical_test(comparison_results_final, wd_vs_we=True) # comparison_results
