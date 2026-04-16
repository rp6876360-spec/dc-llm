import numpy as np
import os
import pandas as pd
import copy
from fin_sentiment.sentiGen import SentimentGeneration

"""
value ordering: original/random/newVar

"""

class DC_RUN:
    def __init__(self, env=None, env_valid=None, **model_kwargs):
        self.env = env # test set env
        self.env_valid = env_valid # valid set env

        self.price_data = model_kwargs['raw_data']

        """
        'varUpdate: 'egenet', 'random'
        'optMode': 'ctrFirst': optimize constraint first, then optimize objective function.
                    'costOpt': optimize cost function, cost = objective value + penalty value.
        """

    def run(self):
        config = self.env.config
        dc_inst = DCAlgo(config=config, env=self.env, price_data=self.price_data) 

        # weight_lst = run_main(env_config=config, env_valid=self.env_valid)
        # print(weight_lst) # (252, 11) with cash, idx=0 is the init.(equal) weightng.
        # print(np.shape(weight_lst), self.env.totalTradeDay)
        
        vpath_bestmodel = os.path.join(config.res_dir, 'valid_bestmodel.csv')
        if os.path.exists(vpath_bestmodel):
            os.rename(vpath_bestmodel, os.path.join(config.res_dir, 'valid_bestmodel_ref.csv'))
        vpath_profile = os.path.join(config.res_dir, 'valid_profile.csv')
        if os.path.exists(vpath_profile):
            os.rename(vpath_profile, os.path.join(config.res_dir, 'valid_profile_ref.csv'))
        vpath_stepdata = os.path.join(config.res_dir, 'valid_stepdata.csv')
        if os.path.exists(vpath_stepdata):
            os.rename(vpath_stepdata, os.path.join(config.res_dir, 'valid_stepdata_ref.csv'))

        obs_state = self.env.reset()
        idxDays = 0
        pf_value = 0

        while True:
            idxDays = idxDays + 1
            if idxDays >= self.env.totalTradeDay:
                curdim = np.shape(weights)[-1]
                weights = np.array([1/curdim]*curdim)
            else:
                weights = dc_inst.run(env=self.env)

            weights = np.reshape(weights, (-1))

            # Check the range of weights outputted from the original algorithm.
            if np.sum(np.abs(weights)) == 0:
                weights = np.array([1/len(weights)]*len(weights)) * self.env.bound_flag
            else:
                weights = weights / np.sum(np.abs(weights))
            weights = np.array([weights])

            obs_state, rewards, terminal_flag, _ = self.env.step(weights)
            
            if terminal_flag:
                break 

            pf_value = self.env.cur_capital

class DCAlgo:
    def __init__(self, config, env, price_data):
        self.config = config
        self.price_data = price_data
        self.grain_mode = self.config.benchmark_algo.split('-')[-1]
        if 'Sentiment' in self.grain_mode:
            self.is_sentiment = True
            self.senti_score = self.read_sentiment()
        else:
            self.is_sentiment = False
            self.senti_score = None

        self.date_lst = np.sort(env.rawdata['date'].unique())

    def read_sentiment(self):
        senti_inst = SentimentGeneration(config=self.config, price_data=self.price_data)
        senti_data = senti_inst.read_sentiment() # [stock, date, score], score [-1, 1]
        return senti_data

    def run(self, env=None):
        
        curday = env.cur_date
        
        refdate_lst = np.sort(self.date_lst[self.date_lst <= curday][-self.config.dc_lookback:])
        
        dc_idx = 1
        cur_data = copy.deepcopy(env.rawdata[env.rawdata['date'].isin(refdate_lst)][['stock', 'date', 'DC-{}'.format(dc_idx)]]) 
        cur_data = cur_data[['stock', 'DC-{}'.format(dc_idx)]].groupby(['stock']).mean().reset_index(drop=False, inplace=False)
        cur_data.sort_values(by=['stock'], ascending=True, inplace=True, ignore_index=True)

        dc_signals = cur_data['DC-{}'.format(dc_idx)].values # range [0, 1]
        
        if self.is_sentiment:
            curdata = copy.deepcopy(self.senti_score[self.senti_score['date']==curday])
            curdata.sort_values(by=['stock'], ascending=True, inplace=True, ignore_index=True)
            curscore = curdata['score'].values # [-1, 1]
            # norm_score
            if self.config.dc_is_norm_score:
                if np.sum(np.abs(curscore)) == 0:
                    curscore = np.array( [1/len(curscore)] * len(curscore) )
                else:
                    curscore = curscore / np.sum(np.abs(curscore))

            weights = dc_signals + (self.config.senti_k * curscore)
            weights = (weights + 1) / 3 # [-1, 2] -> [0, 1]



        else:
            weights = dc_signals        

        if np.sum(np.abs(weights)) == 0:
            weights = np.array([1/len(weights)]*len(weights))
        else:
            weights = weights / np.sum(np.abs(weights))
        
        return weights