import pandas as pd
import numpy as np
import pickle
import torch.nn.functional as F
import torch
from tqdm import tqdm
import warnings

### helper funcs

def create_freq_a(sdc_tnic, min_event=5):
    A_freq = pd.DataFrame(sdc_tnic.AGVKEY.value_counts()).reset_index(drop=False)
    A_freq = A_freq[A_freq.AGVKEY >= min_event]
    A_freq.columns = ["GVKEY", "freq"]
    print(f"totally {A_freq.shape[0]} numbers of frequent Acquirers")
    a_freq_idx_to_gvkey_mapping = {}
    a_freq_gvkey_to_idx_mapping = {}
    for i, row in A_freq.iterrows():
        a_freq_idx_to_gvkey_mapping[i] = row.GVKEY
        a_freq_gvkey_to_idx_mapping[row.GVKEY] = i
    a_freq_lst = A_freq.GVKEY.values.tolist()
    return A_freq,a_freq_lst, a_freq_idx_to_gvkey_mapping, a_freq_gvkey_to_idx_mapping

def same_day_only_one(sdc_tnic_raw):
    print("shape before removing same-day multi events:", sdc_tnic_raw.shape)
    sdc_tnic_raw = sdc_tnic_raw.copy()
    sdc_tnic_one = sdc_tnic_raw.groupby(['AGVKEY', 'DA']).first().reset_index(drop=False)
    print("shape after removing same-day multi events:", sdc_tnic_one.shape)
    sdc_tnic_one.sort_values(by = ['DA'], axis=0, inplace=True)
    return sdc_tnic_one

def minmax_normalize(df):
    '''
    df could be df or array
    '''
    df = df.copy()
    normalized_df=(df-df.mean())/df.std()
    return normalized_df

def minmax_normalize(df):
    df = df.copy()
    normalized_df=(df-df.min())/(df.max()-df.min())
    return normalized_df


def dataloader_preproceser(focal_gvkey):
    
    def get_arr_c(focal_gvkey):
    # part 1, get c  
        def get_focal_df(focal_gvkey):
            '''
            output: will be a df contains 3 columns: DATE, AGVKEY, EVENT_TYPE, SCORE
                DATE: datetime.dt object
                AGVKEY: str: 4 - 6 digits
                EVENT_TYPE: 1:self 0:peer (integer)
                SCORE: TNIC similarity last year for event type 0, otherwise 1
            
            Use DA!!!! not DE

            '''
            def helper1(row):
                if row.AGVKEY == focal_gvkey:
                    return 1 # integer 1
                else:
                    return 0 # integer 0   
            sdc_lst = []
            for focal_year in range(s_year-1, e_year):  
                with open(tmp_data_path+f"/a5_top_10_peers_tnic2_{focal_year}.pickle", 'rb') as f:
                    top_peers = pickle.load(f)
                try:
                    top_peers = top_peers[focal_gvkey] # a dataframe
         #           print(top_peers)
                    top_peers_lst = top_peers.gvkey2.tolist()
                    selected_sdc_tnic = sdc_tnic[ (sdc_tnic['AGVKEY'].isin(top_peers_lst + [focal_gvkey])) & (sdc_tnic.YEAR == focal_year+1) ] 
                    selected_sdc_tnic.reset_index(drop=True)
                    if selected_sdc_tnic.shape[0] > 0:
                        #print(selected_sdc_tnic[['DE', 'AGVKEY']] , top_peers[['gvkey2', 'score']])
                        df = selected_sdc_tnic[['DA', 'AGVKEY', 'TGVKEY']]


                        df['EVENT_TYPE'] = df.apply(helper1, axis=1)
                        #print(df)

                        score_df = top_peers[['gvkey2', 'score']]

                        df = df.merge(score_df, left_on='AGVKEY', right_on = 'gvkey2', how = 'left')
                        df = df[['DA','AGVKEY', 'EVENT_TYPE', 'score', 'TGVKEY']]
        #                print(df)
                        df = df.fillna(1)
                        df.columns = ['UPDATE_DATE','AGVKEY','EVENT_TYPE', 'SCORE', 'TGVKEY'] # rename
                        df = df.reset_index(drop=True)
                        sdc_lst.append(df)
                    #print(len(sdc_lst))
                except:
                    pass

            focal_df = pd.concat(sdc_lst, axis=0)
            focal_c = focal_df.reset_index(drop=True) 
            focal_c = focal_c.sort_values(by = ['UPDATE_DATE']) # date time is unsortable..
            focal_c.reset_index(drop=True, inplace=True)
            return focal_c

        def convert_date(df):
            def datetime_converter(date_time):
                base_time = np.datetime64('1997-01-01')
                days_diff = np.datetime64(date_time.date()) - base_time
                return days_diff.astype(int)
            for idx, row in df.iterrows():
                df.loc[idx, 'UPDATE_DATE'] = datetime_converter(df.loc[idx, 'UPDATE_DATE'])

            df.sort_values(by = ['UPDATE_DATE']).reset_index(drop=True, inplace=True)
            return df

        def making_time_diff(focal_c2):
            '''
            df = focal_c; update date is the integer form that count the date from base_date (1997 01 01)

            WARNING: the No.1 event set time-diff = 0
            '''
            tmp_columns = focal_c2.columns.tolist()
      
            focal_c2['UPDATE_DATE'] = [0] + [1 if timediff==0 else timediff for timediff in focal_c2.UPDATE_DATE.diff().tolist()[1:] ]
            focal_c2.columns = ['time_diff'] + tmp_columns[1:]
            return focal_c2

        def __main__():
            focal_c = get_focal_df(focal_gvkey)
            focal_c2 = convert_date(focal_c.copy())
            focal_c3 = making_time_diff(focal_c2.copy())
            arr_c = np.array(focal_c3[['time_diff', 'EVENT_TYPE', 'SCORE']])
            
            return arr_c, focal_c

        arr, focal_c = __main__()

        return arr, focal_c

    def get_arr_b(focal_c, focal_gvkey):
        def add_datetime(df):
            def helper(row):
                return np.datetime64(str(row.year+1)+'-01-01')
            df['UPDATE_DATE'] = df.apply(helper, axis=1)
            return df
        def obtain_fv(focal_gvkey, focal_c, fv):
            year_min, year_max = min([date.year for date in focal_c.UPDATE_DATE.tolist()]), max([date.year for date in focal_c.UPDATE_DATE.tolist()])
            fv_subset = fv[(fv.year >= year_min-1) & (fv.year <= year_max-1) & (fv.gvkey == focal_gvkey)]
            fv_subset = fv_subset[['gvkey', 'year','UPDATE_DATE', 'at', 'sale', 'ch', 'm2b', 'lev', 'roa', 'ppe',
               'cash2asset', 'cash2sale', 'sale2asset', 'de', 'roe', 'd_sale', 'd_at']]
            fv_subset.columns=['AGVKEY', 'year','UPDATE_DATE', 'at', 'sale', 'ch', 'm2b', 'lev', 'roa', 'ppe',
               'cash2asset', 'cash2sale', 'sale2asset', 'de', 'roe', 'd_sale', 'd_at']
            
            
            return fv_subset

        def __main__():
            with open(tmp_data_path+"/afreq_full_fv.pickle", "rb") as f:
                fv = pickle.load(f)
            fv = add_datetime(fv)
            focal_b = obtain_fv("5047", focal_c, fv)
            arr_b = np.array(focal_b.iloc[:, 3:])
            arr_b = minmax_normalize(arr_b)
            return arr_b, focal_b

        arr = __main__()
        return arr

    def create_main_timeline(focal_b, focal_c):
        '''
        WARNING: GLOBAL and LOCAL time both start from 0!

        '''
        def helper(row):
            if (row.EVENT_TYPE == 1) or (row.EVENT_TYPE == 0):
                return 'past'
            else:
                return "fv"

        def helper2(row):
            if row.EVENT_TYPE == 1:
                return "1"
            elif row.EVENT_TYPE == 0:
                return "2"
            else:
                return "3"

        tmp = pd.concat([focal_c, focal_b]).sort_values(by=['UPDATE_DATE'])
        tmp['EVENT_TYPE_countcreater'] = tmp.apply(helper, axis=1)
        tmp['EVENT_TYPE_true'] = tmp.apply(helper2, axis=1)
        tmp['LOCAL_IDX'] = tmp.groupby(['EVENT_TYPE_countcreater'])['UPDATE_DATE'].rank(ascending=True) -1 # rank start with 1
        tmp['LOCAL_IDX'] = tmp['LOCAL_IDX'].astype(int)
        ## with local_idx using rank, local idx is not continuous (may have gap)
        
        tmp_columns = tmp.columns
        tmp.reset_index(drop=True, inplace=True)
        tmp.reset_index(drop=False, inplace=True)

        tmp.columns = ['GLOBAL_IDX']+ tmp_columns.tolist() # rename global index

        tmp = tmp[['GLOBAL_IDX', 'LOCAL_IDX', 'UPDATE_DATE', 'EVENT_TYPE_true', 'TGVKEY']]

        tmp.columns = ['GLOBAL_IDX', 'LOCAL_IDX', 'UPDATE_DATE', 'EVENT_TYPE', 'TGVKEY'] # rename



        return tmp
    
    def __main__():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            arr_c, focal_c = get_arr_c(focal_gvkey)
        arr_b, focal_b = get_arr_b(focal_c, focal_gvkey)
        timeline = create_main_timeline(focal_b, focal_c)
        return arr_c, arr_b, timeline
    
    arr_c, arr_b, timeline = __main__()
    return arr_c, arr_b, timeline
    







######################################### creating global variables

tmp_data_path = '../MA_data/data/tmp'
data_path = '../MA_data/data'

s_year = 1997
e_year = 2020

load_timeline_from_pickle = False







### run
sdc_tnic = pd.read_pickle(tmp_data_path+f"/sdc_tnic_{s_year}_{e_year}")
sdc_tnic = same_day_only_one(sdc_tnic)
with open(tmp_data_path+f"/tnic_info_3_pairs_{s_year-1}_{e_year-1}", 'rb') as f:
    gvkey_lsts, key_ind_maps , ind_key_maps = pickle.load(f)

A_freq, a_freq_lst, a_freq_idx_to_gvkey_mapping, a_freq_gvkey_to_idx_mapping = create_freq_a(sdc_tnic)

a_freq_info = (A_freq, a_freq_lst, a_freq_idx_to_gvkey_mapping, a_freq_gvkey_to_idx_mapping)

with open(data_path+f"/freq_a_info_{s_year}_{e_year}.pickle", "wb") as f:
    pickle.dump(a_freq_info, f)

if load_timeline_from_pickle:
    with open(data_path+f"/dataset_top10_freq5_{s_year}_{e_year}.pickle", "rb") as f:
        arr_cs, arr_bs, timelines = pickle.load(f)
else:
    arr_cs = []
    arr_bs = []
    timelines = []
    idx_to_gvkey = {}

    for i, gvkey in enumerate(tqdm(a_freq_lst)):
        arr_c, arr_b, timeline = dataloader_preproceser(gvkey)
        
        arr_cs.append(arr_c)
        arr_bs.append(arr_b)
        timelines.append(timeline)
        
        idx_to_gvkey[i] = gvkey
        
    assert len(arr_cs) == len(arr_bs) == len(timelines) == len(a_freq_lst), "length of 3 outputs dismatch with len(a_freq_lst)"

    with open(data_path+f"/dataset_top10_freq5_{s_year}_{e_year}.pickle", "wb") as f:
        pickle.dump((arr_cs, arr_bs, timelines), f)

with open(data_path+"/fv_raw_full_1996_2019.pickle", 'rb') as f:
    fv_full = pickle.load(f)
fv_full = pd.concat([fv_full.iloc[:,:2], minmax_normalize(fv_full.iloc[:, 2:])], axis=1)



######################################## creating variables #######################

    



# creating helpers
def get_arr_b_idx(df):
    '''
    df is the a timeline table of a single firm (the 3rd output of preprocess function)
    output is a list
    
    
    
    '''
    sample_df = df.copy()
    sub1 = sample_df[(sample_df.EVENT_TYPE == '1')& (sample_df.LOCAL_IDX >=2)] # happened at 3rd time 

    global_idxs = sub1.GLOBAL_IDX.values # array

    arr_b_idxs = []
    for global_idx in global_idxs:
        sub2 = sample_df[(sample_df.EVENT_TYPE == '3') & (sample_df.GLOBAL_IDX < global_idx)]
        arr_b_idx = sub2.iloc[-1, 1]
        arr_b_idxs.append(arr_b_idx)

    return arr_b_idxs

def get_c_t_idx(df):
    sample_df = df.copy()
    sub1 = sample_df[(sample_df.EVENT_TYPE == '1')& (sample_df.LOCAL_IDX >=2)] 
    local_idxs = sub1.LOCAL_IDX.values # array
    
    arr_c_idxs = []
    arr_t_idxs = []
    for local_idx in local_idxs:
        sub2 = sample_df[(sample_df.EVENT_TYPE.isin(['1','2'])) & (sample_df.LOCAL_IDX < local_idx)]
        arr_c_idx = sub2.iloc[-1, 1] -1
        arr_t_idx = sub2.iloc[-1, 1]
        arr_c_idxs.append(arr_c_idx)
        arr_t_idxs.append(arr_t_idx)
    return arr_c_idxs, arr_t_idxs

def convert_date(df):
    df = df.copy()
    def datetime_converter(date_time):
        base_time = np.datetime64('1997-01-01')
        days_diff = np.datetime64(date_time.date()) - base_time
        return days_diff.astype(int)
    for idx, row in df.iterrows():
        df.loc[idx, 'UPDATE_DATE_int'] = datetime_converter(df.loc[idx, 'UPDATE_DATE'])

    #df.sort_values(by = ['UPDATE_DATE']).reset_index(drop=True, inplace=True)
    return df 

def sample_negative_time_point(df, base_n_sample=10):
    '''
    df is timeline + 'UPDATE_DATE_int'
    
    number of negative samples is corresponding to the number of positive samples (follow the idea of negative smapling in skip-gram)
        each word, approx 10 negative samples.
    
    '''
    df = df.copy()
    max_time = df.UPDATE_DATE_int.values[-1]
    sub_df = df[df.EVENT_TYPE.isin(['1', '2']) & (df.LOCAL_IDX >=2)]
    min_time = sub_df.UPDATE_DATE_int.values[0]
    n_event = df[(df.EVENT_TYPE == '1') & (df.LOCAL_IDX >=2)].shape[0]
    if n_event == 0:
        n_event = 1
    n_samples = base_n_sample * n_event
    samples = np.random.uniform(low=min_time, high=max_time, size=n_samples)
    return samples, max_time - min_time




def get_arr_b_idx_neg(time_samples, df):
    '''
    df is timeline + 'UPDATE_DATE_int'
    
    '''
    df = df.copy()
    df_b = df[df.EVENT_TYPE == '3']
    arr_b_idxs = []
    for time in time_samples:
        df_b_sub = df_b[df_b.UPDATE_DATE_int<time]
        arr_b_idxs.append(df_b_sub.iloc[-1, 1])
    
    return arr_b_idxs



def get_arr_c_t_idx_neg(time_samples, df):
    '''
    df is timeline + 'UPDATE_DATE_int'
    total columns are: [GLOBAL_IDX  LOCAL_IDX UPDATE_DATE EVENT_TYPE  UPDATE_DATE_int]
    '''
    df = df.copy()
    df_c = df[df.EVENT_TYPE.isin(['1', '2']) & (df.LOCAL_IDX >=2)]

    arr_c_idxs_neg = []
    arr_t_neg = []
    for time in time_samples:
        df_before = df_c[df_c.UPDATE_DATE_int < time]
        #print(time)
        
        arr_c_idx_neg = df_before.iloc[-1, 1] # here do not -1!
        previous_time = df_before.iloc[-1, 5]
        
        arr_c_idxs_neg.append(arr_c_idx_neg)
        #print(time, previous_time)
        arr_t_neg.append(time - previous_time)
    
    return arr_c_idxs_neg, np.array(arr_t_neg)




def get_arr_b_c_idx_i(df, s_year, e_year):
    '''
    df is timeline + 'UPDATE_DATE_int'
    total columns are: [GLOBAL_IDX  LOCAL_IDX UPDATE_DATE EVENT_TYPE  UPDATE_DATE_int]
    '''
    df = df.copy()
    # create a year variable
    def helper(row):
        return row.UPDATE_DATE.year
    df['year'] = df.apply(helper, axis=1)
    
    # qualified self event
    sub = df[(df.EVENT_TYPE == '1') & (df.LOCAL_IDX >= 2)]
    
    yearly = {}
    for year in range(s_year, e_year+1):
        b_idxs = []
        c_idxs = []
        sub2 = sub[sub.year == year] # self event at particular year
        for _, row in sub2.iterrows():
            time = row.UPDATE_DATE_int # float
            # back to global df
            df_b_before = df[(df.UPDATE_DATE_int < time)&(df.EVENT_TYPE == '3')]
            df_c_before = df[(df.UPDATE_DATE_int < time)&(df.EVENT_TYPE.isin(['1','2']))]
            idx_b = df_b_before.iloc[-1, 1]
            idx_c = df_c_before.iloc[-1, 1] -1
            b_idxs.append(idx_b)
            c_idxs.append(idx_c)
            
        yearly[year] = (np.array(b_idxs), np.array(c_idxs))
    
    return yearly
        

def true_tar_idxs_i(timeline, dict_idx):
    '''
    year loop by self-event year
        TNIC related data use year-1
    
    '''

        
    # add year to timeline data
    df = timeline.copy()
    # create a year variable
    def helper(row):
        return row.UPDATE_DATE.year
    df['year'] = df.apply(helper, axis=1)
    
    # qualified self event
    sub = df[(df.EVENT_TYPE == '1') & (df.LOCAL_IDX >= 2)]
    
    yearly = {}
    # loop over self-merge year
    for year in range(s_year, e_year+1):
        '''
        N_i_1 = num of candidate target
        N_i_2 = num of self event
        '''
        N_i_1 = len(gvkey_lsts[year-1]) # all target candidate in TNIC net
        b_idxs, c_idxs = dict_idx[year] # the output of ...
        N_i_2 = len(b_idxs)
        timeline_i = sub[sub.year == year] # only ith year
        targets_lst = timeline_i.TGVKEY.values.tolist() # length = N_i_2
        assert len(targets_lst) == N_i_2, "length dismatch with larget lists and N_i_2"
        idx_lst = [key_ind_maps[year-1][tgvkey] for tgvkey in targets_lst]
        #one_hot_i = (np.arange(_ == a[...,None]-1).astype(int)
        a = np.array(idx_lst)
        one_hot_i = (np.arange(N_i_1) == a[...,None]).astype(int)
        yearly[year] = one_hot_i
    return yearly



def get_node_features(fv_full, gvkey_lsts, key_ind_maps , ind_key_maps, s_year=1997, e_year=2020):
    '''
    fv_full: raw
    gvkey_lsts, key_ind_maps , ind_key_maps: raw
    
    WARNING: the output yearly's year is self-event's year!!! 
    '''
    # loop self-merge year
    yearly = {}
    for year in range(s_year, e_year+1): 
        df_gvkeys = pd.DataFrame({'gvkeys': gvkey_lsts[year-1]})
        fv_candidate = fv_full[fv_full.gvkey.isin(gvkey_lsts[year-1]) & (fv_full.year == year-1)]
        fv_i = df_gvkeys.merge(fv_candidate, left_on='gvkeys', right_on = 'gvkey', how = "left")
        fv_i.reset_index(drop=True, inplace=True)
        #print(fv_i[:5])
        arr = fv_i.iloc[:, 3:].to_numpy()
        yearly[year] = arr
        assert len(gvkey_lsts[year-1]) == arr.shape[0], "list and arr shape dismatch"
    
    return yearly

def get_net_structure(tmp_data_path, gvkey_lsts, key_ind_maps , ind_key_maps, s_year=1997, e_year=2020):
    
    yearly = {}    
    # loop over self-event year! not TNIC !
    for year in range(s_year, e_year+1):     
        with open(tmp_data_path+f'/a5_top_10_peers_tnic2_{year-1}.pickle', 'rb') as f:
            tnic = pickle.load(f)   
            df_all_lst = []
            for _,value in tnic.items():
                df_all_lst.append(value)
            df_all = pd.concat(df_all_lst)
            df_net = df_all[['gvkey1', 'gvkey2']]
            lst1 = df_net.gvkey1.values.tolist()
            lst2 = df_net.gvkey2.values.tolist()
            idx1 = [key_ind_maps[year-1][gvkey1] for gvkey1 in lst1]
            idx2 = [key_ind_maps[year-1][gvkey2] for gvkey2 in lst2]
            arr = np.array([idx1, idx2])
            assert arr.shape[0] == 2, "the dim of output is wrong"
        yearly[year] = arr
    return yearly


######## main helper ####

def create_dataset():
    ma_dataset = []
    for i, gvkey in enumerate(tqdm(a_freq_lst)):
        '''
        the rest variables are for a specific freq acquirer
        
        '''
        
        
    
        ######### get arr_c and arr_delta_time
        arr_c = arr_cs[i]
        arr_c = arr_c[1:] # remove the first row!
        arr_delta_time = arr_c[:, 0]
        # get arr_b
        arr_b = arr_bs[i]
        
        ######## Event data
        timeline = convert_date(timelines[i]) # add UPDATE_DATE_int column
        #print(timeline.columns)
        # arr_b_idx
        arr_b_idx = get_arr_b_idx(timeline)
        arr_b_idx = np.array(arr_b_idx)
        # arr_c_idx and arr_delta_time
        arr_c_idx, arr_t_idx = get_c_t_idx(timeline)
        arr_c_idx = np.array(arr_c_idx)
        arr_t_idx = np.array(arr_t_idx)

        arr_delta_time = arr_delta_time[list(arr_t_idx)]

        event_data = (arr_b_idx, arr_c_idx, arr_delta_time) #############
        
        ######## Non- Event data
        # estimate_time_length
        samples, estimate_time_length = sample_negative_time_point(timeline)
        #  arr_b_idx
        non_arr_b_idx =  get_arr_b_idx_neg(samples, timeline)
        non_arr_b_idx = np.array(non_arr_b_idx)

        non_arr_c_idx, non_arr_delta_time = get_arr_c_t_idx_neg(samples, timeline)
        non_arr_c_idx = np.array(non_arr_c_idx)

        non_event_data = (non_arr_b_idx, non_arr_c_idx, non_arr_delta_time) #############
        ######## Choice data dict
        dict_idx = get_arr_b_c_idx_i(timeline, s_year, e_year)
        true_tar_idxs = true_tar_idxs_i(timeline, dict_idx)
        node_feature = get_node_features(fv_full, gvkey_lsts, key_ind_maps , ind_key_maps, s_year, e_year)
        net_structure = get_net_structure(tmp_data_path, gvkey_lsts, key_ind_maps , ind_key_maps, s_year, e_year)
        choice_data_dict = (dict_idx, true_tar_idxs, node_feature, net_structure)
        ma_dataset.append((arr_b, arr_c, arr_delta_time, event_data, non_event_data, estimate_time_length, choice_data_dict))
  
    return ma_dataset



        


        
        
        
        


################# main ####################

def main():
    # hyperparams

    
    
    ma_dataset = create_dataset()
    with open(data_path+"/ma_dataset_minmax.pickle", 'wb') as f:
        pickle.dump(ma_dataset, f)




main()