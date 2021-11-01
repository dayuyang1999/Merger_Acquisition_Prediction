import pandas as pd
import pickle
import torch
import numpy as np




    

class MADataset:
    def __init__(self):
        self.s_year = 1997
        self.e_year = 2020
        self.read_path = '../MA_data/data'

        # read frequent acuqirer info tuples
        with open(self.read_path+f"/freq_a_info_{self.s_year}_{self.e_year}.pickle", "rb") as f:
            a_freq, a_freq_lst, a_freq_idx_to_gvkey_mapping, a_freq_gvkey_to_idx_mapping = pickle.load(f)
        self.a_freq, self.a_freq_lst, self.a_freq_idx_to_gvkey_mapping, self.a_freq_gvkey_to_idx_mapping = a_freq, a_freq_lst, a_freq_idx_to_gvkey_mapping, a_freq_gvkey_to_idx_mapping

        #read dataset
        with open(self.read_path+f"/ma_dataset_N01.pickle", 'rb') as f: ### 
            ma_dataset = pickle.load(f)
        self.ma_dataset = ma_dataset # a list
        
        assert len(self.ma_dataset) == len(self.a_freq_lst), "a_freq and ma_dataset length dismatch"

    # def load_data(self):
    #         with open(self.read_path+f"/ma_dataset1.pickle", 'rb') as f:
    #             ma_dataset = pickle.load(f)
    #         modified = []
    #         for ele in ma_dataset:
    #             arr_b, arr_c, arr_delta_time, event_data, non_event_data, estimate_length, choice_data_dict = ele
    #             # lst to array
    #             arr_b_idx, arr_c_idx, arr_delta_time = event_data
    #             event_data = np.array(arr_b_idx), np.array(arr_c_idx), arr_delta_time
    #             # lst to array
    #             arr_b_idx, arr_c_idx, arr_delta_time = non_event_data
    #             non_event_data = np.array(arr_b_idx), np.array(arr_c_idx), arr_delta_time
    #             aa, true_tar_idxs, bb, cc = choice_data_dict
    #             new = {}
    #             for k,v in true_tar_idxs.items():
    #                 new[k] = v.to(torch.float).numpy()
    #                 print(type(v))
    #             choice_data_dict = aa, new, bb, cc
    #             modified.append((arr_b, arr_c, arr_delta_time, event_data, non_event_data, estimate_length, choice_data_dict))
    #         return modified
            

    def __getitem__(self, idx):
        return self.ma_dataset[idx]
    
    def __len__(self):
        return len(self.a_freq_lst)

    



class MADataset_test:
    def __init__(self):
        self.s_year = 1997
        self.e_year = 2020
        self.read_path = '../MA_data/data'

        # read frequent acuqirer info tuples
        with open(self.read_path+f"/freq_a_info_{self.s_year}_{self.e_year}.pickle", "rb") as f:
            a_freq, a_freq_lst, a_freq_idx_to_gvkey_mapping, a_freq_gvkey_to_idx_mapping = pickle.load(f)
        self.a_freq, self.a_freq_lst, self.a_freq_idx_to_gvkey_mapping, self.a_freq_gvkey_to_idx_mapping = a_freq, a_freq_lst, a_freq_idx_to_gvkey_mapping, a_freq_gvkey_to_idx_mapping

        #read dataset
        with open(self.read_path+f"/ma_dataset_test.pickle", 'rb') as f: ### 
            ma_dataset = pickle.load(f)
        self.ma_dataset = ma_dataset # a list
        
        #assert len(self.ma_dataset) == len(self.a_freq_lst), "a_freq and ma_dataset length dismatch"

    # def load_data(self):
    #         with open(self.read_path+f"/ma_dataset1.pickle", 'rb') as f:
    #             ma_dataset = pickle.load(f)
    #         modified = []
    #         for ele in ma_dataset:
    #             arr_b, arr_c, arr_delta_time, event_data, non_event_data, estimate_length, choice_data_dict = ele
    #             # lst to array
    #             arr_b_idx, arr_c_idx, arr_delta_time = event_data
    #             event_data = np.array(arr_b_idx), np.array(arr_c_idx), arr_delta_time
    #             # lst to array
    #             arr_b_idx, arr_c_idx, arr_delta_time = non_event_data
    #             non_event_data = np.array(arr_b_idx), np.array(arr_c_idx), arr_delta_time
    #             aa, true_tar_idxs, bb, cc = choice_data_dict
    #             new = {}
    #             for k,v in true_tar_idxs.items():
    #                 new[k] = v.to(torch.float).numpy()
    #                 print(type(v))
    #             choice_data_dict = aa, new, bb, cc
    #             modified.append((arr_b, arr_c, arr_delta_time, event_data, non_event_data, estimate_length, choice_data_dict))
    #         return modified
            

    def __getitem__(self, idx):
        return self.ma_dataset[idx]
    
    def __len__(self):
        return 1 
        


# class MADataset:
#     '''
    
    
    
    
#     '''

#     def __init__(self):
#         ###### basic params no need to tuning
#         self.tmp_data_path = '../MA_data/data/tmp'
#         self.data_path = '../MA_data/data'

#         # WARINING: those are self.event year, not TNIC year
#             # if the focal event is in ith year, use i-1th year TNIC and fv data
#         self.s_year = 1997 
#         self.e_year = 2020

#         # basic data set prepared
#         self.sdc_tnic = self.read_tnic()
#         self.
#         self.idx_to_gvkey_mapping = 

#     @staticmethod
#     def read_tnic(self):
#         sdc_tnic = pd.read_pickle(self.tmp_data_path+f"/sdc_tnic_{self.s_year}_{self.e_year}")
#         return sdc_tnic

    
#     def get_freq_acquirer(self):
#         A_freq = pd.DataFrame(sdc_tnic.AGVKEY.value_counts()).reset_index(drop=False)
#         A_freq = A_freq[A_freq.AGVKEY >= 5]
#         A_freq.columns = ["GVKEY", "freq"]
#         print(f"totally {A_freq.shape[0]} numbers of frequent Acquirers")


#     def get_data_d(self, focal_gvkey):
#         c_d, b_d, timeline_d = self.dataloader_preproceser(focal_gvkey)





#     def __getitem__(self, idx):
#         gvkey = self.idx_to_gvkey_mapping[idx]


