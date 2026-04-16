#！/usr/bin/python
# -*- coding: utf-8 -*-#

'''
---------------------------------
 Name:         snti_analysis.py
 Description:  
 Author:       Samuel
 Date:         28/10/2023
---------------------------------
'''

import pandas as pd
import numpy as np
import os
import copy
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import gc

class FinSentiment:
    def __init__(self):
        self.findata_dir = './fin_sentiment/findata'
        self.sentiment_source = 'multi' # 'benzinga', 'others', 'all' for trans_score, 'multi' for keeping prob of pos, neu, neg.
        self.market_name = 'DJIA' # 'SP500', 'DJIA'
        self.region = 'us'

        self.trading_hour = {
            'us': {'open': '09:30:00', 'close': '16:00:00'},
        }

    def preprocess_data(self):
        fintext_data = pd.read_csv(os.path.join(self.findata_dir, '{}_headlines.csv'.format(self.sentiment_source)), header=0) 
        if 'title' in list(fintext_data.columns):
            fintext_data.rename(columns={'title': 'headline'}, inplace=True)

        # Read index data
        index_weights = pd.read_csv(os.path.join(self.findata_dir, '{}_Weights.csv'.format(self.market_name)), header=0)
        mktidx_code_list = index_weights['code'].tolist()

        selected_data = copy.deepcopy(fintext_data[fintext_data['stock'].isin(mktidx_code_list)][['stock', 'date', 'headline']])
        selected_data.sort_values(by=['stock', 'date'], ascending=True, inplace=True, ignore_index=True)
        selected_data.rename(columns={'date': 'pre_date'}, inplace=True)
        selected_data['pre_date'] = selected_data['pre_date'].apply(lambda x: x[:19])
        selected_data['pre_date'] = pd.to_datetime(selected_data['pre_date'])
        
        if self.sentiment_source == 'benzinga':
            post_date_lst = []
            for idx in range(len(selected_data)):
                curtime = selected_data.loc[idx, 'pre_date']
                # If a news is issued before market close time, we should consider it as the current day's news and use it to predict the adjusted weighting after market close.
                # If a news is issued after market close time, we should consider it as the next day's news and use it to predict the adjusted weighting after next day market close 
                if curtime < pd.Timestamp('{}-{}-{} {}'.format(curtime.year, curtime.month, curtime.day, self.trading_hour[self.region]['close'])): # < 16:00:00
                    post_date = pd.Timestamp('{}-{}-{} 00:00:00'.format(curtime.year, curtime.month, curtime.day))
                else:
                    post_date = pd.Timestamp('{}-{}-{} 00:00:00'.format(curtime.year, curtime.month, curtime.day)) + pd.Timedelta(days=1)
                post_date_lst.append(post_date)
            selected_data['date'] = post_date_lst
        elif self.sentiment_source == 'others':
            selected_data['date'] = selected_data['pre_date'].apply(lambda x: pd.Timestamp('{}-{}-{} 00:00:00'.format(x.year, x.month, x.day)) + pd.Timedelta(days=1))
        else:
            raise ValueError("Wrong sentiment source for preprocessing! [{}]".format(self.sentiment_source))

        # The raw date of benzinga source has considered the EST timezone, i.e., UTC-4 from the 2nd Sunday of March to the 1st Sunday of November. UTC-5 from the 1st Sunday of November to the 2nd Sunday of March next year.
        # We just need to extract the shown date to match with stock

        self.selected_data = copy.deepcopy(selected_data[['stock', 'date', 'headline']])
        self.selected_data.sort_values(by=['stock', 'date'], ascending=True, inplace=True, ignore_index=True)

        del selected_data
        del fintext_data
        gc.collect()

    def sentiment_analysis(self):
        tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        llm_model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        
        input_batch_size = 1000
        # cnt_flag = 18
        for cnt_flag in range(15, 35):
        # if True:
            init_start_idx = cnt_flag * 10000
            num_of_samples = (1+cnt_flag) * 10000

            total_samples = len(self.selected_data) # 339668 (benzinga), 343254 (others)
            if num_of_samples > total_samples:
                num_of_samples = total_samples
            print("current start: {}, end: {}, total: {}".format(init_start_idx, num_of_samples, total_samples))

            start_idx = init_start_idx
            sentiment_score_lst = np.array([]).reshape(-1, 3)
            while True:
                end_idx = start_idx + input_batch_size
                if end_idx > num_of_samples:
                    end_idx = num_of_samples

                batch_headlines = self.selected_data.loc[start_idx: end_idx-1, 'headline'].tolist() # data includes start_idx and (end_idx -1).
                input_ids_lst = tokenizer(batch_headlines, padding=True, truncation=True, return_tensors='pt')
                del batch_headlines
                gc.collect()
                
                outputs = llm_model(**input_ids_lst)
                del input_ids_lst
                gc.collect()

                batch_outputs = outputs.logits.detach().numpy()
                del outputs
                gc.collect()

                sentiment_score_lst = np.append(sentiment_score_lst, batch_outputs, axis=0)
                del batch_outputs
                gc.collect()

                print("Progress: {}/{}..".format(end_idx, num_of_samples))
                start_idx = end_idx
                if start_idx >= num_of_samples:
                    break

            if (num_of_samples-init_start_idx) != sentiment_score_lst.shape[0]:
                raise ValueError("The number of samples [{}] is not equal to the number of sentiment scores [{}]!".format(num_of_samples, sentiment_score_lst.shape))

            sub_selected_data = copy.deepcopy(self.selected_data.loc[init_start_idx: num_of_samples-1, ['stock', 'date', 'headline']])

            sub_selected_data['score_positive'] = sentiment_score_lst[:, 0]
            sub_selected_data['score_neutral'] = sentiment_score_lst[:, 2]
            sub_selected_data['score_negative'] = sentiment_score_lst[:, 1]

            sub_selected_data.sort_values(by=['stock', 'date'], ascending=True, inplace=True, ignore_index=True)
            # The socre is the raw scores, not the normalized/softmax scores.
            sub_selected_data = sub_selected_data[['stock', 'date', 'score_positive', 'score_neutral', 'score_negative', 'headline']]

            subfolder = os.makedirs(os.path.join(self.findata_dir, self.sentiment_source), exist_ok=True)
            sub_selected_data.to_csv(os.path.join(self.findata_dir, self.sentiment_source, '{}_rawSentimentScore_{}.csv'.format(self.sentiment_source, cnt_flag)), index=False)
            
            del sub_selected_data
            gc.collect()

    def merge_files(self):
        # Merge raw scores.
        srcdir = os.path.join(self.findata_dir, self.sentiment_source)

        data = pd.DataFrame()
        for idx in range(34):
            spath = os.path.join(srcdir, '{}_rawSentimentScore_{}.csv'.format(self.sentiment_source, idx))
            subdata = pd.read_csv(spath, header=0)
            data = pd.concat([data, subdata], axis=0, join='outer', ignore_index=True)

            del subdata
            gc.collect()

        data.sort_values(by=['stock', 'date'], ascending=True, inplace=True, ignore_index=True)                     
        data = data[['stock', 'date', 'score_positive', 'score_neutral', 'score_negative', 'headline']]
        data.to_csv(os.path.join(self.findata_dir, '{}_rawSentimentScore_{}.csv'.format(self.sentiment_source, self.market_name)), index=False)
        print("Done merge")

    def trans_score(self):
        # Transform the raw scores [pos, neu, neg] to a sentiment score.
        # raw_score -> softmax -> avg prob -> high class -> pos[+x], neu[0], neg[-x] 
        # ['stock', 'date', 'score_positive', 'score_neutral', 'score_negative', 'headline']

        index_weights = pd.read_csv(os.path.join(self.findata_dir, '{}_Weights.csv'.format(self.market_name)), header=0, usecols=['code', 'rank'])
        index_weights.rename(columns={'code': 'stock'}, inplace=True)
        index_weights.sort_values(by=['rank'], ascending=True, inplace=True, ignore_index=True)
        mktidx_stock_lst = index_weights['stock'].unique().tolist()

        data = pd.DataFrame()
        if (self.sentiment_source == 'benzinga') or (self.sentiment_source == 'others'):
            num_of_files = len(os.listdir(os.path.join(self.findata_dir, self.sentiment_source)))
            for idx in range(num_of_files):
                spath = os.path.join(self.findata_dir, self.sentiment_source, '{}_rawSentimentScore_{}.csv'.format(self.sentiment_source, idx))
                allsubdata = pd.read_csv(spath, header=0, usecols=['stock', 'date', 'score_positive', 'score_neutral', 'score_negative'])
                
                subdata = copy.deepcopy(allsubdata[allsubdata['stock'].isin(mktidx_stock_lst)])
                del allsubdata
                gc.collect()
                subdata.sort_values(by=['stock', 'date'], ascending=True, inplace=True, ignore_index=True)

                subdata['date'] = pd.to_datetime(subdata['date'])
                score_np = torch.nn.functional.softmax(torch.from_numpy(np.array(subdata[['score_positive', 'score_neutral', 'score_negative']])), dim=-1).numpy()
                subdata['score_positive'] = score_np[:, 0] # after softmax
                subdata['score_neutral'] = score_np[:, 1]
                subdata['score_negative'] = score_np[:, 2]
                
                data = pd.concat([data, subdata], axis=0, join='outer', ignore_index=True)
                del subdata
                gc.collect()
        elif self.sentiment_source in ['all', 'multi']:
            for src in ['benzinga', 'others']:
                num_of_files = len(os.listdir(os.path.join(self.findata_dir, src)))
                for idx in range(num_of_files):
                    spath = os.path.join(self.findata_dir, src, '{}_rawSentimentScore_{}.csv'.format(src, idx))
                    allsubdata = pd.read_csv(spath, header=0, usecols=['stock', 'date', 'score_positive', 'score_neutral', 'score_negative'])

                    subdata = copy.deepcopy(allsubdata[allsubdata['stock'].isin(mktidx_stock_lst)])
                    del allsubdata
                    gc.collect()
                    subdata.sort_values(by=['stock', 'date'], ascending=True, inplace=True, ignore_index=True)

                    subdata['date'] = pd.to_datetime(subdata['date'])
                    score_np = torch.nn.functional.softmax(torch.from_numpy(np.array(subdata[['score_positive', 'score_neutral', 'score_negative']])), dim=-1).numpy()
                    subdata['score_positive'] = score_np[:, 0] # after softmax
                    subdata['score_neutral'] = score_np[:, 1]
                    subdata['score_negative'] = score_np[:, 2]
                    if self.sentiment_source == 'multi':
                        subdata['source'] = src
                    data = pd.concat([data, subdata], axis=0, join='outer', ignore_index=True)
                    del subdata
                    gc.collect()
        else:
            raise ValueError("Wrong sentiment source for transforming! [{}]".format(self.sentiment_source))

        if self.sentiment_source == 'multi':
            data = data.groupby(['stock', 'date', 'source']).mean().reset_index(drop=False, inplace=False)

        else:
            data = data.groupby(['stock', 'date']).mean().reset_index(drop=False, inplace=False)
            tscore = np.zeros(len(data))

            slst = np.array(data[['score_positive', 'score_neutral', 'score_negative']])
            flag_max = np.argmax(slst, axis=1)
            idxpos = np.argwhere(flag_max == 0).flatten()
            idxneg = np.argwhere(flag_max == 2).flatten()
            
            tscore[idxpos] = slst[idxpos, 0]
            tscore[idxneg] = (-1) * slst[idxneg, 2]
            data['score'] = tscore

        # stock name -> rank symbol
        merge_data = pd.merge(data, index_weights, on=['stock'], how='left')

        if np.sum(merge_data['rank'].isnull()) > 0:
            nulldata = merge_data[merge_data['rank'].isnull()]['stock'].unique()
            print("Check Null: {}, list: {}".format(len(nulldata), nulldata))

        # check
        check_mkt = index_weights['stock'].tolist()
        check_sent = data['stock'].unique().tolist()
        # Sentiment has the stock-x, but the stock-x does not exist in the market index.
        extra1 = list(set(check_sent) - set(check_mkt))
        if len(extra1) > 0:
            print("Sentiment-Y, mkt-N: {}, list: {}".format(len(extra1), np.sort(extra1)))
        extra2 = list(set(check_mkt) - set(check_sent))
        if len(extra2) > 0:
            extra21 = index_weights[index_weights['stock'].isin(extra2)][['rank', 'stock']]
            extra21.sort_values(by=['rank'], ascending=True, inplace=True, ignore_index=True)
            print("Sentiment-N, mkt-Y: {}, list: \n{}".format(len(extra2), np.array(extra21)))
            print(np.sort(extra21['rank'].values))

        if self.sentiment_source == 'multi':
            merge_data = merge_data[['rank', 'date', 'score_positive', 'score_neutral', 'score_negative', 'source']] # stock is rank_idx here
            merge_data.rename(columns={'rank': 'stock'}, inplace=True)
            merge_data.sort_values(by=['stock', 'date', 'source'], ascending=True, inplace=True, ignore_index=True)        
        else:
            merge_data = merge_data[['rank', 'date', 'score']] # stock is rank_idx here
            merge_data.rename(columns={'rank': 'stock'}, inplace=True)
            merge_data.sort_values(by=['stock', 'date'], ascending=True, inplace=True, ignore_index=True)
        merge_data.to_csv(os.path.join(self.findata_dir, '{}_SentimentScore_{}.csv'.format(self.sentiment_source, self.market_name)), index=False)
        

        #
        datelst = np.sort(np.array(merge_data['date'].unique()))
        print("start date: {}, end date: {}".format(datelst[0], datelst[-1]))

    def run(self):
        # self.preprocess_data()
        # self.sentiment_analysis()

        self.trans_score()

        """
        two source -> raw_score [positive, neutral, negative]
        raw score -> sentiment score
        sentiment score -> RL
        """
def main():
    fin_inst = FinSentiment()
    fin_inst.run()


if __name__ == '__main__':
    main()
