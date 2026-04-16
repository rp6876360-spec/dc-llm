#！/usr/bin/python
# -*- coding: utf-8 -*-#

'''
---------------------------------
 Name:         sentiGen.py
 Description:  Read and Generate sentiment score 
 Author:       Samuel
---------------------------------
'''
import os
import numpy as np
import pandas as pd
import copy
from typing import List, Union
from scipy import spatial
import gc

class SentimentGeneration:
    def __init__(self, config, price_data):
        self.config = config
        self.date_lst = pd.to_datetime(price_data['date'].unique().tolist()) # pd.Timetstamp format
        self.stock_lst = price_data['stock'].unique()
        self.price_data = price_data
        grain_mode = self.config.benchmark_algo.split('-')[-1]
        self.read_sentiment = self.read_sentiment_avg
        if grain_mode == 'SentimentBenzinga':
            self.senti_source = 'benzinga'
        elif grain_mode == 'SentimentNonBenzinga':
            self.senti_source = 'others'
        elif 'SentimentMulti' in grain_mode:
            self.senti_source = 'all' # benzinga, all
            if 'Corr' in grain_mode:
                self.read_sentiment = self.read_sentiment_corr
        else:
            raise ValueError('Unknown grain mode: {}'.format(grain_mode))
        self.findata_dir = './fin_sentiment/findata'

    def read_sentiment_avg(self):
        spath = os.path.join(self.findata_dir, '{}_SentimentScore_{}.csv'.format(self.senti_source, self.config.market_name))
        raw_senti_data = pd.read_csv(spath, header=0)
        # [stock, date, score], score [-1, 1]
        # select the stock
        senti_data = copy.deepcopy(raw_senti_data[raw_senti_data['stock'].isin(self.stock_lst)])
        del raw_senti_data
        gc.collect()

        senti_data['date'] = pd.to_datetime(senti_data['date'])
        # fill_missing_dates
        fill_data = pd.DataFrame()
        senti_stock_lst = senti_data['stock'].unique()
        for sigStock in senti_stock_lst:
            exist_date = senti_data[senti_data['stock'] == sigStock]['date'].tolist()
            miss_date = list(set(list(self.date_lst)) - set(exist_date))
            sigFill = pd.DataFrame({'date': miss_date})
            sigFill['stock'] = sigStock
            sigFill['score'] = 0.0
            fill_data = pd.concat([fill_data, sigFill], axis=0, join='outer', ignore_index=True)
        
        senti_data = pd.concat([senti_data, fill_data], axis=0, join='outer', ignore_index=True)
        # remove the date that is not in the traidng period
        senti_data = senti_data[(senti_data['date'] >= self.config.train_date_start) & (senti_data['date'] <= self.config.test_date_end)]
        
        score = senti_data['score'].values
        invalid_value = score[(score < (-1)) | (score > 1)]
        if len(invalid_value) > 0:
            raise ValueError('Invalid sentiment score value [{}]: {}'.format(len(invalid_value), invalid_value))

        miss_stocks = set(list(self.stock_lst)) - set(senti_data['stock'].unique().tolist())
        if len(miss_stocks) > 0:
            raise ValueError('Missing stocks in sentiment data [{}]: {}'.format(len(miss_stocks), miss_stocks))

        senti_data.sort_values(by=['stock', 'date'], ascending=True, inplace=True, ignore_index=True)
        # [stock, date, score], score [-1, 1]
        return senti_data


    def read_sentiment_corr(self):
        spath = os.path.join(self.findata_dir, 'multi_SentimentScore_{}.csv'.format(self.config.market_name))
        raw_senti_data = pd.read_csv(spath, header=0)
        senti_data = copy.deepcopy(raw_senti_data[raw_senti_data['stock'].isin(self.stock_lst)])
        del raw_senti_data
        gc.collect()

        senti_data['date'] = pd.to_datetime(senti_data['date'])
        # fill_missing_dates
        fill_data = pd.DataFrame()
        senti_stock_lst = senti_data['stock'].unique()
        senti_src_lst = senti_data['source'].unique()
        
        price_date_lst = self.price_data['date'].unique()
        all_date_lst = np.sort(price_date_lst)
        price_date_lst = np.sort(price_date_lst[(price_date_lst >= self.config.train_date_start) & (price_date_lst <= self.config.test_date_end)])
        
        senti_data['cnt'] = 1
        union_date_pd = senti_data[['stock', 'date', 'cnt']].groupby(['stock','date']).sum().reset_index(drop=False, inplace=False)
        union_date_pd = copy.deepcopy(union_date_pd[union_date_pd['cnt']==2][['date', 'stock']])
        union_date_pd.reset_index(drop=True, inplace=True)

        default_corr = 0.0 
        score_pd = {'date': [], 'source': [], 'stock': [], 'weighted_score': [], 'corr': [], 'raw_score': []} 
        for idx1, src in enumerate(senti_src_lst):
            # if src != 'others': # benzinga, others
            #     continue
            for idx2, sigstock in enumerate(senti_stock_lst):
                for sigDate in price_date_lst:
                    score_pd['date'].append(sigDate)
                    score_pd['source'].append(src)
                    score_pd['stock'].append(sigstock)

                    cur_raw_score = senti_data[(senti_data['date'] == sigDate) & (senti_data['stock'] == sigstock) & (senti_data['source'] == src)][['score_positive', 'score_neutral', 'score_negative']]
                    if len(cur_raw_score) == 0:
                        score_pd['weighted_score'].append(0)
                        score_pd['corr'].append(-1)
                        score_pd['raw_score'].append(0)
                        continue
                    elif len(cur_raw_score) == 1:
                        cur_raw_score = cur_raw_score.values[0] # pos, neu, neg
                        maxidx = np.argmax(cur_raw_score)
                        if maxidx == 0:
                            cur_postscore = cur_raw_score[0]
                        elif maxidx == 1:
                            cur_postscore = 0.0
                        elif maxidx == 2:
                            cur_postscore = - cur_raw_score[2]
                        else:
                            raise ValueError('Unexpected maxidx: {}'.format(maxidx))

                    else:
                        raise ValueError('Unexpected score length: {}'.format(len(cur_raw_score)))

                    score_pd['raw_score'].append(cur_postscore)

                    lb_dates = np.sort(all_date_lst[all_date_lst<sigDate])[-self.config.senti_corr_lookback:]
                    lbscores = copy.deepcopy(senti_data[(senti_data['date'].isin(lb_dates)) & (senti_data['stock'] == sigstock) & (senti_data['source'] == src)][['date', 'score_positive', 'score_neutral', 'score_negative']])
                    if len(lbscores) == 0:
                        score_pd['weighted_score'].append(default_corr * cur_postscore)
                        score_pd['corr'].append(default_corr)
                        continue
    
                    lbscores.sort_values(by=['date'], ascending=True, inplace=True, ignore_index=True)

                    include_dates = np.append(lb_dates, [sigDate], axis=0)
                    cur_price = copy.deepcopy(self.price_data[(self.price_data['date']<=sigDate) & (self.price_data['date']>=include_dates[0]) & (self.price_data['stock'] == sigstock)][['date', 'close']])
                    cur_price.sort_values(by=['date'], ascending=True, inplace=True, ignore_index=True)
                    c_ay = cur_price['close'].values
                    c_pct = (c_ay[1:] / c_ay[:-1]) - 1
                    # pos, neu, neg                    
                    pos_idx = np.argwhere(c_pct>0).flatten()
                    neu_idx = np.argwhere(c_pct==0).flatten()
                    neg_idx = np.argwhere(c_pct<0).flatten()
                    onehot_labels = np.zeros((len(c_pct), 3))
                    onehot_labels[pos_idx, 0] = 1
                    onehot_labels[neu_idx, 1] = 1
                    onehot_labels[neg_idx, 2] = 1
                    onehot_pd = pd.DataFrame(onehot_labels, columns=['pos', 'neu', 'neg'])
                    onehot_pd['date'] = cur_price['date'].values[:-1]
                    exist_news_dates = np.sort(lbscores['date'].unique())
                    onehot_pd = onehot_pd[onehot_pd['date'].isin(exist_news_dates)]
                    onehot_pd.sort_values(by=['date'], ascending=True, inplace=True, ignore_index=True)
                    s_prob = lbscores[['score_positive', 'score_neutral', 'score_negative']].values
                    p_label = onehot_pd[['pos', 'neu', 'neg']].values
                    
                    # Cross entropy
                    dist_perdate = np.sum((-1) * p_label * np.log(s_prob), axis=1) # distance per date, cross-entropy
                    corr_val = np.mean(dist_perdate)
                    if corr_val > 1:
                        corr_val = 1.0
                    
                    # M1
                    corr_val = 1 - corr_val # [0, 1]

                    score_pd['weighted_score'].append(corr_val * cur_postscore) # [0, 1] * [-1, 1] -> [-1, 1]
                    score_pd['corr'].append(corr_val)

        
        score_pd = pd.DataFrame(score_pd)
        # sum-weighted score, range: [-num_of_sources， num_of_sources]
        if self.config.senti_corr_combine == 'sum':
            score_pd = score_pd[['date', 'stock', 'weighted_score']].groupby(['date', 'stock']).sum().reset_index(drop=False, inplace=False)
            score_pd.rename(columns={'weighted_score': 'score'}, inplace=True)
            # score_pd['score'] = (score_pd['score'].values) / len(senti_src_lst) # [-1, 1]
        elif self.config.senti_corr_combine == 'corrmax':
            new_pd = {'date': [], 'stock': [], 'score': []}
            for sigstock in score_pd.stock.unique():
                for sigDate in score_pd.date.unique():
                    cur_score_pd = copy.deepcopy(score_pd[(score_pd['date'] == sigDate) & (score_pd['stock'] == sigstock)])
                    if len(cur_score_pd) == 0:
                        raise
                    elif len(cur_score_pd) == 1:
                        new_pd['date'].append(sigDate)
                        new_pd['stock'].append(sigstock)
                        new_pd['score'].append(cur_score_pd['weighted_score'].values[0])
                    elif len(cur_score_pd) == 2:
                        new_pd['date'].append(sigDate)
                        new_pd['stock'].append(sigstock)
                        corr_lst = cur_score_pd['corr'].values
                        score_lst = cur_score_pd['weighted_score'].values
                        if corr_lst[0] > corr_lst[1]:
                            midx = 0
                        elif corr_lst[0] < corr_lst[1]:
                            midx = 1
                        else:
                            midx = np.random.randint(0, 2)
                        
                        if corr_lst[midx] != 0:
                            new_pd['score'].append(score_lst[midx]/corr_lst[midx])
                        else:
                            new_pd['score'].append(score_lst[midx])
                    else:
                        raise
            new_pd = pd.DataFrame(new_pd)
            score_pd = new_pd
        else:
            raise ValueError('Unknown combine mode: {}'.format(self.config.senti_corr_combine))

        tx = score_pd['score'].values
        txn = tx[tx<(-1)]
        txp = tx[tx>1]
        print("txn: {}, txp: {}".format(len(txn), len(txp)))
        print("Finish reading sentiment data..")
        return score_pd


