#！/usr/bin/python
# -*- coding: utf-8 -*-#
'''
---------------------------------
 Name: utils.py  
 Author: RiPO
--------------------------------
'''
import numpy as np
import os
import pandas as pd
import copy

def post_process(config):
    # post-process after training
    fdir = os.path.join('./res', 'tmp_{}'.format(config.tmp_name))
    os.makedirs(fdir, exist_ok=True)
    # Extract the annuralized return, MDD, sharpe ratio, final capital from the profile files.
    if config.mode == 'Benchmark':
        correspond_profile = pd.DataFrame(pd.read_csv(os.path.join(config.res_dir, 'test_profile.csv'), header=0))
        step_data = pd.DataFrame(pd.read_csv(os.path.join(config.res_dir, 'test_stepdata.csv'), header=0))
        step_data_col = list(step_data.columns)
        shortTermRisk_name = 'risk_policy_validbest'
        if shortTermRisk_name not in step_data_col:
            shortTermRisk_name = 'risk_policy_best'
        if len(correspond_profile) != 1:
            raise ValueError("[Benchmark] the record in test profile may be over than 1 [has {}].".format(len(correspond_profile)))

        test_cputime = np.mean(correspond_profile['cputime'].values)
        test_wall_clock_time = np.mean(correspond_profile['systime'].values)
        train_cputime = 0
        train_wall_clock_time = 0
        valid_cputime = 0
        valid_wall_clock_time = 0

    elif (config.mode == 'RLcontroller') or (config.mode == 'RLonly'):
        valid_best_file = pd.DataFrame(pd.read_csv(os.path.join(config.res_dir, 'valid_bestmodel.csv'), header=0))
        best_ep = int(valid_best_file['{}_ep'.format(config.trained_best_model_type)].values[0])
        test_profile = pd.DataFrame(pd.read_csv(os.path.join(config.res_dir, 'test_profile.csv'), header=0))
        test_cputime = np.mean(test_profile['cputime'].values)
        test_wall_clock_time = np.mean(test_profile['systime'].values)
        correspond_profile = test_profile[test_profile['ep'] == best_ep]
        step_data = pd.DataFrame(pd.read_csv(os.path.join(config.res_dir, 'test_stepdata.csv'), header=0))
        step_data_col = list(step_data.columns)
        shortTermRisk_name = 'risk_policy_validbest' 

        train_profile = pd.DataFrame(pd.read_csv(os.path.join(config.res_dir, 'train_profile.csv'), header=0))
        train_cputime = np.mean(train_profile['cputime'].values)
        train_wall_clock_time = np.mean(train_profile['systime'].values)
        valid_profile = pd.DataFrame(pd.read_csv(os.path.join(config.res_dir, 'valid_profile.csv'), header=0))
        valid_cputime = np.mean(valid_profile['cputime'].values)
        valid_wall_clock_time = np.mean(valid_profile['systime'].values)
    else:
        raise ValueError('Unexpected mode {}'.format(config.mode))
    final_capital_corr = float(correspond_profile['final_capital'].values[0])
    ar_corr = float(correspond_profile['annualReturn_pct'].values[0])
    mdd_corr = float(correspond_profile['mdd'].values[0])
    sr_corr = float(correspond_profile['sharpeRatio'].values[0])
    downsideVol_corr = float(correspond_profile['risk_downsideAtVol'].values[0])
    vol_corr = float(correspond_profile['volatility'].values[0])
    shortTermRisk = np.mean(np.array(step_data[shortTermRisk_name].values))
    # check wether the path exists.
    # info: mode, market, period_mode, rl_model, stock_num, 
    # Record the current date, final capital, AR, MDD, SR
    sigdata = {'current_date': [config.cur_datetime], 'final_capital': [final_capital_corr], 'annualReturn_pct': [ar_corr], 'mdd': [mdd_corr], 
                'sharpeRatio': [sr_corr], 'short-termRisk': [shortTermRisk], 'downsideVolatility': [downsideVol_corr], 'volatility': [vol_corr], 
                'seed': [config.seed_num], 'train_cputime': [train_cputime], 'train_wall_clock_time': [train_wall_clock_time], 
                'valid_cputime': [valid_cputime], 'valid_wall_clock_time': [valid_wall_clock_time], 'test_cputime': [test_cputime], 'test_wall_clock_time': [test_wall_clock_time]} 
    fname = '{}_{}_{}_{}_{}_{}.csv'.format(config.mode, config.market_name, config.period_mode, config.benchmark_algo, config.topK, config.is_enable_dynamic_risk_bound)
    fpath = os.path.join(fdir, fname)
    if os.path.exists(fpath):
        rec_data = pd.DataFrame(pd.read_csv(fpath, header=0))
        # merge dataframes
        rec_data = pd.concat([rec_data, pd.DataFrame(sigdata)], axis=0, join='outer', ignore_index=True)
        rec_data.sort_values(by=['seed'], ascending=True, inplace=True)
        rec_data.reset_index(drop=True, inplace=True)
    else:
        rec_data = pd.DataFrame(sigdata)
    rec_data = copy.deepcopy(rec_data[['current_date', 'final_capital', 'annualReturn_pct', 'mdd', 'sharpeRatio', 'short-termRisk', 'downsideVolatility', 'volatility', 'seed', 
                'train_cputime', 'train_wall_clock_time', 'valid_cputime', 'valid_wall_clock_time', 'test_cputime', 'test_wall_clock_time']])
    rec_data.to_csv(fpath, index=False)

    # calculate the average/std/max/min/median of the final capital, AR, MDD, SR, the folder name(current date) of exp list.
    date_lst = str(rec_data['current_date'].values.tolist())
    fc_avg = rec_data['final_capital'].mean()
    fc_std = rec_data['final_capital'].std() # ddof=1
    fc_best = rec_data['final_capital'].max()
    fc_worst = rec_data['final_capital'].min()
    fc_median = rec_data['final_capital'].median()

    ar_avg = rec_data['annualReturn_pct'].mean()
    ar_std = rec_data['annualReturn_pct'].std() # ddof=1
    ar_best = rec_data['annualReturn_pct'].max()
    ar_worst = rec_data['annualReturn_pct'].min()
    ar_median = rec_data['annualReturn_pct'].median()

    mdd_avg = rec_data['mdd'].mean()
    mdd_std = rec_data['mdd'].std() # ddof=1
    mdd_best = rec_data['mdd'].min()
    mdd_worst = rec_data['mdd'].max()
    mdd_median = rec_data['mdd'].median()

    sr_avg = rec_data['sharpeRatio'].mean()
    sr_std = rec_data['sharpeRatio'].std() # ddof=1
    sr_best = rec_data['sharpeRatio'].max()
    sr_worst = rec_data['sharpeRatio'].min()
    sr_median = rec_data['sharpeRatio'].median()

    downvol_avg = rec_data['downsideVolatility'].mean()
    downvol_std = rec_data['downsideVolatility'].std() # ddof=1
    downvol_best = rec_data['downsideVolatility'].min()
    downvol_worst = rec_data['downsideVolatility'].max()
    downvol_median = rec_data['downsideVolatility'].median()

    vol_avg = rec_data['volatility'].mean()
    vol_std = rec_data['volatility'].std() # ddof=1
    vol_best = rec_data['volatility'].min()
    vol_worst = rec_data['volatility'].max()
    vol_median = rec_data['volatility'].median()

    strisk_avg = rec_data['short-termRisk'].mean()
    strisk_std = rec_data['short-termRisk'].std() # ddof=1
    strisk_best = rec_data['short-termRisk'].min()
    strisk_worst = rec_data['short-termRisk'].max()
    strisk_median = rec_data['short-termRisk'].median()

    train_cputime_avg = rec_data['train_cputime'].mean()
    train_cputime_std = rec_data['train_cputime'].std() # ddof=1
    train_cputime_best = rec_data['train_cputime'].min()
    train_cputime_worst = rec_data['train_cputime'].max()
    train_cputime_median = rec_data['train_cputime'].median()

    train_wall_clock_time_avg = rec_data['train_wall_clock_time'].mean()
    train_wall_clock_time_std = rec_data['train_wall_clock_time'].std() # ddof=1
    train_wall_clock_time_best = rec_data['train_wall_clock_time'].min()
    train_wall_clock_time_worst = rec_data['train_wall_clock_time'].max()
    train_wall_clock_time_median = rec_data['train_wall_clock_time'].median()

    valid_cputime_avg = rec_data['valid_cputime'].mean()
    valid_cputime_std = rec_data['valid_cputime'].std() # ddof=1
    valid_cputime_best = rec_data['valid_cputime'].min()
    valid_cputime_worst = rec_data['valid_cputime'].max()
    valid_cputime_median = rec_data['valid_cputime'].median()

    valid_wall_clock_time_avg = rec_data['valid_wall_clock_time'].mean()
    valid_wall_clock_time_std = rec_data['valid_wall_clock_time'].std() # ddof=1
    valid_wall_clock_time_best = rec_data['valid_wall_clock_time'].min()
    valid_wall_clock_time_worst = rec_data['valid_wall_clock_time'].max()
    valid_wall_clock_time_median = rec_data['valid_wall_clock_time'].median()

    test_cputime_avg = rec_data['test_cputime'].mean()
    test_cputime_std = rec_data['test_cputime'].std() # ddof=1
    test_cputime_best = rec_data['test_cputime'].min()
    test_cputime_worst = rec_data['test_cputime'].max()
    test_cputime_median = rec_data['test_cputime'].median()

    test_wall_clock_time_avg = rec_data['test_wall_clock_time'].mean()
    test_wall_clock_time_std = rec_data['test_wall_clock_time'].std() # ddof=1
    test_wall_clock_time_best = rec_data['test_wall_clock_time'].min()
    test_wall_clock_time_worst = rec_data['test_wall_clock_time'].max()
    test_wall_clock_time_median = rec_data['test_wall_clock_time'].median()
    
    num_of_cur_runs = len(rec_data)
    fpath = os.path.join('./res', 'summary_ref_{}.csv'.format(config.tmp_name)) # Just for reference, not accurate when conducting the hyper-para/ablation experiments as all those results will be summarized in the same record.
    if os.path.exists(fpath):
        summary_data = pd.DataFrame(pd.read_csv(fpath, header=0))
        sig_summarydata = summary_data[(summary_data['mode']==config.mode) & (summary_data['market_name']==config.market_name) & (summary_data['period_mode']==config.period_mode) & (summary_data['algorithm']==config.benchmark_algo) & (summary_data['stock_num']==config.topK) 
                                    & (summary_data['ars']==config.is_enable_dynamic_risk_bound)]
        if len(sig_summarydata) == 0:
            sig_summarydata = {'num_of_runs': [num_of_cur_runs], 'mode': [config.mode], 'market_name': [config.market_name], 'period_mode': [config.period_mode], 'algorithm': [config.benchmark_algo], 'stock_num': [config.topK], 'ars': [config.is_enable_dynamic_risk_bound],
                            'final_capital_avg': [fc_avg], 'final_capital_std': [fc_std], 'final_capital_best': [fc_best], 'final_capital_worst': [fc_worst], 'final_capital_median': [fc_median],
                            'annualReturn_pct_avg': [ar_avg], 'annualReturn_pct_std': [ar_std], 'annualReturn_pct_best': [ar_best], 'annualReturn_pct_worst': [ar_worst], 'annualReturn_pct_median': [ar_median],
                            'mdd_avg': [mdd_avg], 'mdd_std': [mdd_std], 'mdd_best': [mdd_best], 'mdd_worst': [mdd_worst], 'mdd_median': [mdd_median],
                            'sharpeRatio_avg': [sr_avg], 'sharpeRatio_std': [sr_std], 'sharpeRatio_best': [sr_best], 'sharpeRatio_worst': [sr_worst], 'sharpeRatio_median': [sr_median],
                            'short-termRisk_avg': [strisk_avg], 'short-termRisk_std': [strisk_std], 'short-termRisk_best': [strisk_best], 'short-termRisk_worst': [strisk_worst], 'short-termRisk_median': [strisk_median],
                            'downsideVolatility_avg': [downvol_avg], 'downsideVolatility_std': [downvol_std], 'downsideVolatility_best': [downvol_best], 'downsideVolatility_worst': [downvol_worst], 'downsideVolatility_median': [downvol_median],
                            'volatility_avg': [vol_avg], 'volatility_std': [vol_std], 'volatility_best': [vol_best], 'volatility_worst': [vol_worst], 'volatility_median': [vol_median],
                            'train_cputime_avg': [train_cputime_avg], 'train_cputime_std': [train_cputime_std], 'train_cputime_best': [train_cputime_best], 'train_cputime_worst': [train_cputime_worst], 'train_cputime_median': [train_cputime_median],
                            'train_wall_clock_time_avg': [train_wall_clock_time_avg], 'train_wall_clock_time_std': [train_wall_clock_time_std], 'train_wall_clock_time_best': [train_wall_clock_time_best], 'train_wall_clock_time_worst': [train_wall_clock_time_worst], 'train_wall_clock_time_median': [train_wall_clock_time_median],
                            'valid_cputime_avg': [valid_cputime_avg], 'valid_cputime_std': [valid_cputime_std], 'valid_cputime_best': [valid_cputime_best], 'valid_cputime_worst': [valid_cputime_worst], 'valid_cputime_median': [valid_cputime_median],
                            'valid_wall_clock_time_avg': [valid_wall_clock_time_avg], 'valid_wall_clock_time_std': [valid_wall_clock_time_std], 'valid_wall_clock_time_best': [valid_wall_clock_time_best], 'valid_wall_clock_time_worst': [valid_wall_clock_time_worst], 'valid_wall_clock_time_median': [valid_wall_clock_time_median],
                            'test_cputime_avg': [test_cputime_avg], 'test_cputime_std': [test_cputime_std], 'test_cputime_best': [test_cputime_best], 'test_cputime_worst': [test_cputime_worst], 'test_cputime_median': [test_cputime_median],
                            'test_wall_clock_time_avg': [test_wall_clock_time_avg], 'test_wall_clock_time_std': [test_wall_clock_time_std], 'test_wall_clock_time_best': [test_wall_clock_time_best], 'test_wall_clock_time_worst': [test_wall_clock_time_worst], 'test_wall_clock_time_median': [test_wall_clock_time_median],
                            'date_list': [date_lst]}
            sig_summarydata = pd.DataFrame(sig_summarydata)
            summary_data = pd.concat([summary_data, sig_summarydata], axis=0, join='outer', ignore_index=True)

        elif len(sig_summarydata) == 1:
            summary_data.loc[(summary_data['mode']==config.mode) & (summary_data['market_name']==config.market_name) & (summary_data['period_mode']==config.period_mode) & (summary_data['algorithm']==config.benchmark_algo) & (summary_data['stock_num']==config.topK) & (summary_data['ars']==config.is_enable_dynamic_risk_bound), 'num_of_runs'] = num_of_cur_runs
            summary_data.loc[(summary_data['mode']==config.mode) & (summary_data['market_name']==config.market_name) & (summary_data['period_mode']==config.period_mode) & (summary_data['algorithm']==config.benchmark_algo) & (summary_data['stock_num']==config.topK) & (summary_data['ars']==config.is_enable_dynamic_risk_bound), 'final_capital_avg'] = fc_avg
            summary_data.loc[(summary_data['mode']==config.mode) & (summary_data['market_name']==config.market_name) & (summary_data['period_mode']==config.period_mode) & (summary_data['algorithm']==config.benchmark_algo) & (summary_data['stock_num']==config.topK) & (summary_data['ars']==config.is_enable_dynamic_risk_bound), 'final_capital_std'] = fc_std
            summary_data.loc[(summary_data['mode']==config.mode) & (summary_data['market_name']==config.market_name) & (summary_data['period_mode']==config.period_mode) & (summary_data['algorithm']==config.benchmark_algo) & (summary_data['stock_num']==config.topK) & (summary_data['ars']==config.is_enable_dynamic_risk_bound), 'final_capital_best'] = fc_best
            summary_data.loc[(summary_data['mode']==config.mode) & (summary_data['market_name']==config.market_name) & (summary_data['period_mode']==config.period_mode) & (summary_data['algorithm']==config.benchmark_algo) & (summary_data['stock_num']==config.topK) & (summary_data['ars']==config.is_enable_dynamic_risk_bound), 'final_capital_worst'] = fc_worst
            summary_data.loc[(summary_data['mode']==config.mode) & (summary_data['market_name']==config.market_name) & (summary_data['period_mode']==config.period_mode) & (summary_data['algorithm']==config.benchmark_algo) & (summary_data['stock_num']==config.topK) & (summary_data['ars']==config.is_enable_dynamic_risk_bound), 'final_capital_median'] = fc_median

            summary_data.loc[(summary_data['mode']==config.mode) & (summary_data['market_name']==config.market_name) & (summary_data['period_mode']==config.period_mode) & (summary_data['algorithm']==config.benchmark_algo) & (summary_data['stock_num']==config.topK) & (summary_data['ars']==config.is_enable_dynamic_risk_bound), 'annualReturn_pct_avg'] = ar_avg
            summary_data.loc[(summary_data['mode']==config.mode) & (summary_data['market_name']==config.market_name) & (summary_data['period_mode']==config.period_mode) & (summary_data['algorithm']==config.benchmark_algo) & (summary_data['stock_num']==config.topK) & (summary_data['ars']==config.is_enable_dynamic_risk_bound), 'annualReturn_pct_std'] = ar_std
            summary_data.loc[(summary_data['mode']==config.mode) & (summary_data['market_name']==config.market_name) & (summary_data['period_mode']==config.period_mode) & (summary_data['algorithm']==config.benchmark_algo) & (summary_data['stock_num']==config.topK) & (summary_data['ars']==config.is_enable_dynamic_risk_bound), 'annualReturn_pct_best'] = ar_best
            summary_data.loc[(summary_data['mode']==config.mode) & (summary_data['market_name']==config.market_name) & (summary_data['period_mode']==config.period_mode) & (summary_data['algorithm']==config.benchmark_algo) & (summary_data['stock_num']==config.topK) & (summary_data['ars']==config.is_enable_dynamic_risk_bound), 'annualReturn_pct_worst'] = ar_worst
            summary_data.loc[(summary_data['mode']==config.mode) & (summary_data['market_name']==config.market_name) & (summary_data['period_mode']==config.period_mode) & (summary_data['algorithm']==config.benchmark_algo) & (summary_data['stock_num']==config.topK) & (summary_data['ars']==config.is_enable_dynamic_risk_bound), 'annualReturn_pct_median'] = ar_median

            summary_data.loc[(summary_data['mode']==config.mode) & (summary_data['market_name']==config.market_name) & (summary_data['period_mode']==config.period_mode) & (summary_data['algorithm']==config.benchmark_algo) & (summary_data['stock_num']==config.topK) & (summary_data['ars']==config.is_enable_dynamic_risk_bound), 'mdd_avg'] = mdd_avg
            summary_data.loc[(summary_data['mode']==config.mode) & (summary_data['market_name']==config.market_name) & (summary_data['period_mode']==config.period_mode) & (summary_data['algorithm']==config.benchmark_algo) & (summary_data['stock_num']==config.topK) & (summary_data['ars']==config.is_enable_dynamic_risk_bound), 'mdd_std'] = mdd_std
            summary_data.loc[(summary_data['mode']==config.mode) & (summary_data['market_name']==config.market_name) & (summary_data['period_mode']==config.period_mode) & (summary_data['algorithm']==config.benchmark_algo) & (summary_data['stock_num']==config.topK) & (summary_data['ars']==config.is_enable_dynamic_risk_bound), 'mdd_best'] = mdd_best
            summary_data.loc[(summary_data['mode']==config.mode) & (summary_data['market_name']==config.market_name) & (summary_data['period_mode']==config.period_mode) & (summary_data['algorithm']==config.benchmark_algo) & (summary_data['stock_num']==config.topK) & (summary_data['ars']==config.is_enable_dynamic_risk_bound), 'mdd_worst'] = mdd_worst
            summary_data.loc[(summary_data['mode']==config.mode) & (summary_data['market_name']==config.market_name) & (summary_data['period_mode']==config.period_mode) & (summary_data['algorithm']==config.benchmark_algo) & (summary_data['stock_num']==config.topK) & (summary_data['ars']==config.is_enable_dynamic_risk_bound), 'mdd_median'] = mdd_median

            summary_data.loc[(summary_data['mode']==config.mode) & (summary_data['market_name']==config.market_name) & (summary_data['period_mode']==config.period_mode) & (summary_data['algorithm']==config.benchmark_algo) & (summary_data['stock_num']==config.topK) & (summary_data['ars']==config.is_enable_dynamic_risk_bound), 'sharpeRatio_avg'] = sr_avg
            summary_data.loc[(summary_data['mode']==config.mode) & (summary_data['market_name']==config.market_name) & (summary_data['period_mode']==config.period_mode) & (summary_data['algorithm']==config.benchmark_algo) & (summary_data['stock_num']==config.topK) & (summary_data['ars']==config.is_enable_dynamic_risk_bound), 'sharpeRatio_std'] = sr_std
            summary_data.loc[(summary_data['mode']==config.mode) & (summary_data['market_name']==config.market_name) & (summary_data['period_mode']==config.period_mode) & (summary_data['algorithm']==config.benchmark_algo) & (summary_data['stock_num']==config.topK) & (summary_data['ars']==config.is_enable_dynamic_risk_bound), 'sharpeRatio_best'] = sr_best
            summary_data.loc[(summary_data['mode']==config.mode) & (summary_data['market_name']==config.market_name) & (summary_data['period_mode']==config.period_mode) & (summary_data['algorithm']==config.benchmark_algo) & (summary_data['stock_num']==config.topK) & (summary_data['ars']==config.is_enable_dynamic_risk_bound), 'sharpeRatio_worst'] = sr_worst
            summary_data.loc[(summary_data['mode']==config.mode) & (summary_data['market_name']==config.market_name) & (summary_data['period_mode']==config.period_mode) & (summary_data['algorithm']==config.benchmark_algo) & (summary_data['stock_num']==config.topK) & (summary_data['ars']==config.is_enable_dynamic_risk_bound), 'sharpeRatio_median'] = sr_median

            summary_data.loc[(summary_data['mode']==config.mode) & (summary_data['market_name']==config.market_name) & (summary_data['period_mode']==config.period_mode) & (summary_data['algorithm']==config.benchmark_algo) & (summary_data['stock_num']==config.topK) & (summary_data['ars']==config.is_enable_dynamic_risk_bound), 'short-termRisk_avg'] = strisk_avg
            summary_data.loc[(summary_data['mode']==config.mode) & (summary_data['market_name']==config.market_name) & (summary_data['period_mode']==config.period_mode) & (summary_data['algorithm']==config.benchmark_algo) & (summary_data['stock_num']==config.topK) & (summary_data['ars']==config.is_enable_dynamic_risk_bound), 'short-termRisk_std'] = strisk_std
            summary_data.loc[(summary_data['mode']==config.mode) & (summary_data['market_name']==config.market_name) & (summary_data['period_mode']==config.period_mode) & (summary_data['algorithm']==config.benchmark_algo) & (summary_data['stock_num']==config.topK) & (summary_data['ars']==config.is_enable_dynamic_risk_bound), 'short-termRisk_best'] = strisk_best
            summary_data.loc[(summary_data['mode']==config.mode) & (summary_data['market_name']==config.market_name) & (summary_data['period_mode']==config.period_mode) & (summary_data['algorithm']==config.benchmark_algo) & (summary_data['stock_num']==config.topK) & (summary_data['ars']==config.is_enable_dynamic_risk_bound), 'short-termRisk_worst'] = strisk_worst
            summary_data.loc[(summary_data['mode']==config.mode) & (summary_data['market_name']==config.market_name) & (summary_data['period_mode']==config.period_mode) & (summary_data['algorithm']==config.benchmark_algo) & (summary_data['stock_num']==config.topK) & (summary_data['ars']==config.is_enable_dynamic_risk_bound), 'short-termRisk_median'] = strisk_median

            summary_data.loc[(summary_data['mode']==config.mode) & (summary_data['market_name']==config.market_name) & (summary_data['period_mode']==config.period_mode) & (summary_data['algorithm']==config.benchmark_algo) & (summary_data['stock_num']==config.topK) & (summary_data['ars']==config.is_enable_dynamic_risk_bound), 'downsideVolatility_avg'] = downvol_avg
            summary_data.loc[(summary_data['mode']==config.mode) & (summary_data['market_name']==config.market_name) & (summary_data['period_mode']==config.period_mode) & (summary_data['algorithm']==config.benchmark_algo) & (summary_data['stock_num']==config.topK) & (summary_data['ars']==config.is_enable_dynamic_risk_bound), 'downsideVolatility_std'] = downvol_std
            summary_data.loc[(summary_data['mode']==config.mode) & (summary_data['market_name']==config.market_name) & (summary_data['period_mode']==config.period_mode) & (summary_data['algorithm']==config.benchmark_algo) & (summary_data['stock_num']==config.topK) & (summary_data['ars']==config.is_enable_dynamic_risk_bound), 'downsideVolatility_best'] = downvol_best
            summary_data.loc[(summary_data['mode']==config.mode) & (summary_data['market_name']==config.market_name) & (summary_data['period_mode']==config.period_mode) & (summary_data['algorithm']==config.benchmark_algo) & (summary_data['stock_num']==config.topK) & (summary_data['ars']==config.is_enable_dynamic_risk_bound), 'downsideVolatility_worst'] = downvol_worst
            summary_data.loc[(summary_data['mode']==config.mode) & (summary_data['market_name']==config.market_name) & (summary_data['period_mode']==config.period_mode) & (summary_data['algorithm']==config.benchmark_algo) & (summary_data['stock_num']==config.topK) & (summary_data['ars']==config.is_enable_dynamic_risk_bound), 'downsideVolatility_median'] = downvol_median

            summary_data.loc[(summary_data['mode']==config.mode) & (summary_data['market_name']==config.market_name) & (summary_data['period_mode']==config.period_mode) & (summary_data['algorithm']==config.benchmark_algo) & (summary_data['stock_num']==config.topK) & (summary_data['ars']==config.is_enable_dynamic_risk_bound), 'volatility_avg'] = vol_avg
            summary_data.loc[(summary_data['mode']==config.mode) & (summary_data['market_name']==config.market_name) & (summary_data['period_mode']==config.period_mode) & (summary_data['algorithm']==config.benchmark_algo) & (summary_data['stock_num']==config.topK) & (summary_data['ars']==config.is_enable_dynamic_risk_bound), 'volatility_std'] = vol_std
            summary_data.loc[(summary_data['mode']==config.mode) & (summary_data['market_name']==config.market_name) & (summary_data['period_mode']==config.period_mode) & (summary_data['algorithm']==config.benchmark_algo) & (summary_data['stock_num']==config.topK) & (summary_data['ars']==config.is_enable_dynamic_risk_bound), 'volatility_best'] = vol_best
            summary_data.loc[(summary_data['mode']==config.mode) & (summary_data['market_name']==config.market_name) & (summary_data['period_mode']==config.period_mode) & (summary_data['algorithm']==config.benchmark_algo) & (summary_data['stock_num']==config.topK) & (summary_data['ars']==config.is_enable_dynamic_risk_bound), 'volatility_worst'] = vol_worst
            summary_data.loc[(summary_data['mode']==config.mode) & (summary_data['market_name']==config.market_name) & (summary_data['period_mode']==config.period_mode) & (summary_data['algorithm']==config.benchmark_algo) & (summary_data['stock_num']==config.topK) & (summary_data['ars']==config.is_enable_dynamic_risk_bound), 'volatility_median'] = vol_median

            summary_data.loc[(summary_data['mode']==config.mode) & (summary_data['market_name']==config.market_name) & (summary_data['period_mode']==config.period_mode) & (summary_data['algorithm']==config.benchmark_algo) & (summary_data['stock_num']==config.topK) & (summary_data['ars']==config.is_enable_dynamic_risk_bound), 'train_cputime_avg'] = train_cputime_avg
            summary_data.loc[(summary_data['mode']==config.mode) & (summary_data['market_name']==config.market_name) & (summary_data['period_mode']==config.period_mode) & (summary_data['algorithm']==config.benchmark_algo) & (summary_data['stock_num']==config.topK) & (summary_data['ars']==config.is_enable_dynamic_risk_bound), 'train_cputime_std'] = train_cputime_std
            summary_data.loc[(summary_data['mode']==config.mode) & (summary_data['market_name']==config.market_name) & (summary_data['period_mode']==config.period_mode) & (summary_data['algorithm']==config.benchmark_algo) & (summary_data['stock_num']==config.topK) & (summary_data['ars']==config.is_enable_dynamic_risk_bound), 'train_cputime_best'] = train_cputime_best
            summary_data.loc[(summary_data['mode']==config.mode) & (summary_data['market_name']==config.market_name) & (summary_data['period_mode']==config.period_mode) & (summary_data['algorithm']==config.benchmark_algo) & (summary_data['stock_num']==config.topK) & (summary_data['ars']==config.is_enable_dynamic_risk_bound), 'train_cputime_worst'] = train_cputime_worst
            summary_data.loc[(summary_data['mode']==config.mode) & (summary_data['market_name']==config.market_name) & (summary_data['period_mode']==config.period_mode) & (summary_data['algorithm']==config.benchmark_algo) & (summary_data['stock_num']==config.topK) & (summary_data['ars']==config.is_enable_dynamic_risk_bound), 'train_cputime_median'] = train_cputime_median

            summary_data.loc[(summary_data['mode']==config.mode) & (summary_data['market_name']==config.market_name) & (summary_data['period_mode']==config.period_mode) & (summary_data['algorithm']==config.benchmark_algo) & (summary_data['stock_num']==config.topK) & (summary_data['ars']==config.is_enable_dynamic_risk_bound), 'train_wall_clock_time_avg'] = train_wall_clock_time_avg
            summary_data.loc[(summary_data['mode']==config.mode) & (summary_data['market_name']==config.market_name) & (summary_data['period_mode']==config.period_mode) & (summary_data['algorithm']==config.benchmark_algo) & (summary_data['stock_num']==config.topK) & (summary_data['ars']==config.is_enable_dynamic_risk_bound), 'train_wall_clock_time_std'] = train_wall_clock_time_std
            summary_data.loc[(summary_data['mode']==config.mode) & (summary_data['market_name']==config.market_name) & (summary_data['period_mode']==config.period_mode) & (summary_data['algorithm']==config.benchmark_algo) & (summary_data['stock_num']==config.topK) & (summary_data['ars']==config.is_enable_dynamic_risk_bound), 'train_wall_clock_time_best'] = train_wall_clock_time_best
            summary_data.loc[(summary_data['mode']==config.mode) & (summary_data['market_name']==config.market_name) & (summary_data['period_mode']==config.period_mode) & (summary_data['algorithm']==config.benchmark_algo) & (summary_data['stock_num']==config.topK) & (summary_data['ars']==config.is_enable_dynamic_risk_bound), 'train_wall_clock_time_worst'] = train_wall_clock_time_worst
            summary_data.loc[(summary_data['mode']==config.mode) & (summary_data['market_name']==config.market_name) & (summary_data['period_mode']==config.period_mode) & (summary_data['algorithm']==config.benchmark_algo) & (summary_data['stock_num']==config.topK) & (summary_data['ars']==config.is_enable_dynamic_risk_bound), 'train_wall_clock_time_median'] = train_wall_clock_time_median

            summary_data.loc[(summary_data['mode']==config.mode) & (summary_data['market_name']==config.market_name) & (summary_data['period_mode']==config.period_mode) & (summary_data['algorithm']==config.benchmark_algo) & (summary_data['stock_num']==config.topK) & (summary_data['ars']==config.is_enable_dynamic_risk_bound), 'valid_cputime_avg'] = valid_cputime_avg
            summary_data.loc[(summary_data['mode']==config.mode) & (summary_data['market_name']==config.market_name) & (summary_data['period_mode']==config.period_mode) & (summary_data['algorithm']==config.benchmark_algo) & (summary_data['stock_num']==config.topK) & (summary_data['ars']==config.is_enable_dynamic_risk_bound), 'valid_cputime_std'] = valid_cputime_std
            summary_data.loc[(summary_data['mode']==config.mode) & (summary_data['market_name']==config.market_name) & (summary_data['period_mode']==config.period_mode) & (summary_data['algorithm']==config.benchmark_algo) & (summary_data['stock_num']==config.topK) & (summary_data['ars']==config.is_enable_dynamic_risk_bound), 'valid_cputime_best'] = valid_cputime_best
            summary_data.loc[(summary_data['mode']==config.mode) & (summary_data['market_name']==config.market_name) & (summary_data['period_mode']==config.period_mode) & (summary_data['algorithm']==config.benchmark_algo) & (summary_data['stock_num']==config.topK) & (summary_data['ars']==config.is_enable_dynamic_risk_bound), 'valid_cputime_worst'] = valid_cputime_worst
            summary_data.loc[(summary_data['mode']==config.mode) & (summary_data['market_name']==config.market_name) & (summary_data['period_mode']==config.period_mode) & (summary_data['algorithm']==config.benchmark_algo) & (summary_data['stock_num']==config.topK) & (summary_data['ars']==config.is_enable_dynamic_risk_bound), 'valid_cputime_median'] = valid_cputime_median

            summary_data.loc[(summary_data['mode']==config.mode) & (summary_data['market_name']==config.market_name) & (summary_data['period_mode']==config.period_mode) & (summary_data['algorithm']==config.benchmark_algo) & (summary_data['stock_num']==config.topK) & (summary_data['ars']==config.is_enable_dynamic_risk_bound), 'valid_wall_clock_time_avg'] = valid_wall_clock_time_avg
            summary_data.loc[(summary_data['mode']==config.mode) & (summary_data['market_name']==config.market_name) & (summary_data['period_mode']==config.period_mode) & (summary_data['algorithm']==config.benchmark_algo) & (summary_data['stock_num']==config.topK) & (summary_data['ars']==config.is_enable_dynamic_risk_bound), 'valid_wall_clock_time_std'] = valid_wall_clock_time_std
            summary_data.loc[(summary_data['mode']==config.mode) & (summary_data['market_name']==config.market_name) & (summary_data['period_mode']==config.period_mode) & (summary_data['algorithm']==config.benchmark_algo) & (summary_data['stock_num']==config.topK) & (summary_data['ars']==config.is_enable_dynamic_risk_bound), 'valid_wall_clock_time_best'] = valid_wall_clock_time_best
            summary_data.loc[(summary_data['mode']==config.mode) & (summary_data['market_name']==config.market_name) & (summary_data['period_mode']==config.period_mode) & (summary_data['algorithm']==config.benchmark_algo) & (summary_data['stock_num']==config.topK) & (summary_data['ars']==config.is_enable_dynamic_risk_bound), 'valid_wall_clock_time_worst'] = valid_wall_clock_time_worst
            summary_data.loc[(summary_data['mode']==config.mode) & (summary_data['market_name']==config.market_name) & (summary_data['period_mode']==config.period_mode) & (summary_data['algorithm']==config.benchmark_algo) & (summary_data['stock_num']==config.topK) & (summary_data['ars']==config.is_enable_dynamic_risk_bound), 'valid_wall_clock_time_median'] = valid_wall_clock_time_median

            summary_data.loc[(summary_data['mode']==config.mode) & (summary_data['market_name']==config.market_name) & (summary_data['period_mode']==config.period_mode) & (summary_data['algorithm']==config.benchmark_algo) & (summary_data['stock_num']==config.topK) & (summary_data['ars']==config.is_enable_dynamic_risk_bound), 'test_cputime_avg'] = test_cputime_avg
            summary_data.loc[(summary_data['mode']==config.mode) & (summary_data['market_name']==config.market_name) & (summary_data['period_mode']==config.period_mode) & (summary_data['algorithm']==config.benchmark_algo) & (summary_data['stock_num']==config.topK) & (summary_data['ars']==config.is_enable_dynamic_risk_bound), 'test_cputime_std'] = test_cputime_std
            summary_data.loc[(summary_data['mode']==config.mode) & (summary_data['market_name']==config.market_name) & (summary_data['period_mode']==config.period_mode) & (summary_data['algorithm']==config.benchmark_algo) & (summary_data['stock_num']==config.topK) & (summary_data['ars']==config.is_enable_dynamic_risk_bound), 'test_cputime_best'] = test_cputime_best
            summary_data.loc[(summary_data['mode']==config.mode) & (summary_data['market_name']==config.market_name) & (summary_data['period_mode']==config.period_mode) & (summary_data['algorithm']==config.benchmark_algo) & (summary_data['stock_num']==config.topK) & (summary_data['ars']==config.is_enable_dynamic_risk_bound), 'test_cputime_worst'] = test_cputime_worst
            summary_data.loc[(summary_data['mode']==config.mode) & (summary_data['market_name']==config.market_name) & (summary_data['period_mode']==config.period_mode) & (summary_data['algorithm']==config.benchmark_algo) & (summary_data['stock_num']==config.topK) & (summary_data['ars']==config.is_enable_dynamic_risk_bound), 'test_cputime_median'] = test_cputime_median

            summary_data.loc[(summary_data['mode']==config.mode) & (summary_data['market_name']==config.market_name) & (summary_data['period_mode']==config.period_mode) & (summary_data['algorithm']==config.benchmark_algo) & (summary_data['stock_num']==config.topK) & (summary_data['ars']==config.is_enable_dynamic_risk_bound), 'test_wall_clock_time_avg'] = test_wall_clock_time_avg
            summary_data.loc[(summary_data['mode']==config.mode) & (summary_data['market_name']==config.market_name) & (summary_data['period_mode']==config.period_mode) & (summary_data['algorithm']==config.benchmark_algo) & (summary_data['stock_num']==config.topK) & (summary_data['ars']==config.is_enable_dynamic_risk_bound), 'test_wall_clock_time_std'] = test_wall_clock_time_std
            summary_data.loc[(summary_data['mode']==config.mode) & (summary_data['market_name']==config.market_name) & (summary_data['period_mode']==config.period_mode) & (summary_data['algorithm']==config.benchmark_algo) & (summary_data['stock_num']==config.topK) & (summary_data['ars']==config.is_enable_dynamic_risk_bound), 'test_wall_clock_time_best'] = test_wall_clock_time_best
            summary_data.loc[(summary_data['mode']==config.mode) & (summary_data['market_name']==config.market_name) & (summary_data['period_mode']==config.period_mode) & (summary_data['algorithm']==config.benchmark_algo) & (summary_data['stock_num']==config.topK) & (summary_data['ars']==config.is_enable_dynamic_risk_bound), 'test_wall_clock_time_worst'] = test_wall_clock_time_worst
            summary_data.loc[(summary_data['mode']==config.mode) & (summary_data['market_name']==config.market_name) & (summary_data['period_mode']==config.period_mode) & (summary_data['algorithm']==config.benchmark_algo) & (summary_data['stock_num']==config.topK) & (summary_data['ars']==config.is_enable_dynamic_risk_bound), 'test_wall_clock_time_median'] = test_wall_clock_time_median

            summary_data.loc[(summary_data['mode']==config.mode) & (summary_data['market_name']==config.market_name) & (summary_data['period_mode']==config.period_mode) & (summary_data['algorithm']==config.benchmark_algo) & (summary_data['stock_num']==config.topK) & (summary_data['ars']==config.is_enable_dynamic_risk_bound), 'date_list'] = date_lst

        else:
            raise ValueError('Unexpected length of sig_summarydata {}'.format(len(sig_summarydata)))

    else:
        sig_summarydata ={'num_of_runs': [num_of_cur_runs], 'mode': [config.mode], 'market_name': [config.market_name], 'period_mode': [config.period_mode], 'algorithm': [config.benchmark_algo], 'stock_num': [config.topK], 'ars': [config.is_enable_dynamic_risk_bound],
                            'final_capital_avg': [fc_avg], 'final_capital_std': [fc_std], 'final_capital_best': [fc_best], 'final_capital_worst': [fc_worst], 'final_capital_median': [fc_median],
                            'annualReturn_pct_avg': [ar_avg], 'annualReturn_pct_std': [ar_std], 'annualReturn_pct_best': [ar_best], 'annualReturn_pct_worst': [ar_worst], 'annualReturn_pct_median': [ar_median],
                            'mdd_avg': [mdd_avg], 'mdd_std': [mdd_std], 'mdd_best': [mdd_best], 'mdd_worst': [mdd_worst], 'mdd_median': [mdd_median],
                            'sharpeRatio_avg': [sr_avg], 'sharpeRatio_std': [sr_std], 'sharpeRatio_best': [sr_best], 'sharpeRatio_worst': [sr_worst], 'sharpeRatio_median': [sr_median],
                            'short-termRisk_avg': [strisk_avg], 'short-termRisk_std': [strisk_std], 'short-termRisk_best': [strisk_best], 'short-termRisk_worst': [strisk_worst], 'short-termRisk_median': [strisk_median],
                            'downsideVolatility_avg': [downvol_avg], 'downsideVolatility_std': [downvol_std], 'downsideVolatility_best': [downvol_best], 'downsideVolatility_worst': [downvol_worst], 'downsideVolatility_median': [downvol_median],
                            'volatility_avg': [vol_avg], 'volatility_std': [vol_std], 'volatility_best': [vol_best], 'volatility_worst': [vol_worst], 'volatility_median': [vol_median],
                            'train_cputime_avg': [train_cputime_avg], 'train_cputime_std': [train_cputime_std], 'train_cputime_best': [train_cputime_best], 'train_cputime_worst': [train_cputime_worst], 'train_cputime_median': [train_cputime_median],
                            'train_wall_clock_time_avg': [train_wall_clock_time_avg], 'train_wall_clock_time_std': [train_wall_clock_time_std], 'train_wall_clock_time_best': [train_wall_clock_time_best], 'train_wall_clock_time_worst': [train_wall_clock_time_worst], 'train_wall_clock_time_median': [train_wall_clock_time_median],
                            'valid_cputime_avg': [valid_cputime_avg], 'valid_cputime_std': [valid_cputime_std], 'valid_cputime_best': [valid_cputime_best], 'valid_cputime_worst': [valid_cputime_worst], 'valid_cputime_median': [valid_cputime_median],
                            'valid_wall_clock_time_avg': [valid_wall_clock_time_avg], 'valid_wall_clock_time_std': [valid_wall_clock_time_std], 'valid_wall_clock_time_best': [valid_wall_clock_time_best], 'valid_wall_clock_time_worst': [valid_wall_clock_time_worst], 'valid_wall_clock_time_median': [valid_wall_clock_time_median],
                            'test_cputime_avg': [test_cputime_avg], 'test_cputime_std': [test_cputime_std], 'test_cputime_best': [test_cputime_best], 'test_cputime_worst': [test_cputime_worst], 'test_cputime_median': [test_cputime_median],
                            'test_wall_clock_time_avg': [test_wall_clock_time_avg], 'test_wall_clock_time_std': [test_wall_clock_time_std], 'test_wall_clock_time_best': [test_wall_clock_time_best], 'test_wall_clock_time_worst': [test_wall_clock_time_worst], 'test_wall_clock_time_median': [test_wall_clock_time_median],
                            'date_list': [date_lst]}
        summary_data = pd.DataFrame(sig_summarydata)
    summary_data = summary_data[['num_of_runs', 'mode', 'market_name', 'period_mode', 'algorithm', 'stock_num', 'ars',
                                    'annualReturn_pct_avg', 'annualReturn_pct_std', 'mdd_avg', 'mdd_std', 'sharpeRatio_avg', 'sharpeRatio_std',
                                    'short-termRisk_avg', 'short-termRisk_std', 
                                    'train_cputime_avg', 'train_wall_clock_time_avg', 'valid_cputime_avg', 'valid_wall_clock_time_avg', 'test_cputime_avg', 'test_wall_clock_time_avg',
                                    'downsideVolatility_avg', 'downsideVolatility_std', 'volatility_avg', 'volatility_std', 'final_capital_avg', 'final_capital_std',
                                    'annualReturn_pct_best', 'annualReturn_pct_worst', 'annualReturn_pct_median', 'mdd_best', 'mdd_worst', 'mdd_median', 'sharpeRatio_best', 'sharpeRatio_worst', 'sharpeRatio_median', 
                                    'short-termRisk_best', 'short-termRisk_worst', 'short-termRisk_median', 'downsideVolatility_best', 'downsideVolatility_worst', 'downsideVolatility_median', 'volatility_best', 'volatility_worst', 'volatility_median', 'final_capital_best', 'final_capital_worst', 'final_capital_median', 
                                    'train_cputime_std', 'train_cputime_best', 'train_cputime_worst', 'train_cputime_median', 'train_wall_clock_time_std', 'train_wall_clock_time_best', 'train_wall_clock_time_worst', 'train_wall_clock_time_median',
                                    'valid_cputime_std', 'valid_cputime_best', 'valid_cputime_worst', 'valid_cputime_median', 'valid_wall_clock_time_std', 'valid_wall_clock_time_best', 'valid_wall_clock_time_worst', 'valid_wall_clock_time_median',
                                    'test_cputime_std', 'test_cputime_best', 'test_cputime_worst', 'test_cputime_median', 'test_wall_clock_time_std', 'test_wall_clock_time_best', 'test_wall_clock_time_worst', 'test_wall_clock_time_median',
                                    'date_list']]
    summary_data.to_csv(fpath, index=False)

    print("Done [post-process]")

