# Border-Theory
A Computational Linguistic Approach to Study Border Theory at Scale

Public repository containing the data and code for running the work-life analysis in "A Computational Linguistic Approach to Study Border Theory at Scale" (CSCW'24).

## Instructions
There are two main files in this repository: topic_modelling.py and work_life_analysis.py. These files must be ran in order.
1. Run topic_modelling.py on the anonymized Twitter data (Journalists.csv used as example). Multiple files will be created during this run:
    * 'unprocessed_tweets_journalists_test_filt.csv': A csv of tweets for users who
        have total tweets >= specified threshold. These users are known as filtered users.
    * 'unprocessed_journalists_tweets_active_members.csv': A csv of tweets for filtered users
        who tweet in each period under study: 2019, 2020, 2021, 2022. These are known as active users.
    * 'unprocessed_journalists_tweets_active_members*.csv': A csv of tweets for active users
        who are determined to be journalists.
    * Journalists dir: Stores extracted nouns data.
    * pickles dir: Stores pickled files used as input into the BERTopic model.
    * BERTopic dir: Stores pickled BERTopic output files for analysis in work_life_analysis.py.
2. Run work_life_analysis.py to produce the results shown in the paper. **NB**: You will have to manually inspect the BERTopic topics before grouping them into work or life categories (as described on line 1271 of script, '3. Define topic labels'). Multiple files will be created during this run:
    * 'Journalists_swb_all_lockdowns.pdf': The subjective well-being plots for journalists.
    * 'Journalists_flex_all_lockdowns.pdf': The flexibility plots for journalists.
    * 'norm_frequency_comparison_plot_Journalists_lockdowns_across_lockdowns.pdf': The normalized
        frequency plots for journalists during lockdown.
    * 'norm_frequency_comparison_plot_Journalists_baselines_across_lockdowns.pdf': The normalized
        frequency plots for journalists during baseline period.
## Requirements for code
The requirements (with python >= 3.9) can be found [here](https://github.com/TimDouglas28/Border-Theory/blob/main/requirements.txt)
