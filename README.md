# Predicting Merger and Acquisition: A Deep Learning Method


This Project is my first summer project in my Ph.D. program. 

# Introduction
Merger and Acquisition(MA) is contagious, a.k.a, previous events can effect the the future event. Most traditional literature use "acquisition likelihood model" (based on logistic regression) to predict MA. There are some disavantages:
- Can only predict one side of the deal (however, obviously, MA is a mutual aggrement)
    - if making deal-level prediction, must use resampling techniques to alleviate unbalanced data

I first formulate MA predictive task as a link prediction task. Then I propose a MA predictive model based on Temporal Dynamic Industry Network (TDIN). 
TDIN models the dynamics over network using temporal point process. The network structure is pre-defined though financial discosures.

---

The project can be seperated into 2 subpart:
- data processing
- modeling

In many Deep Learning task, data is pretty easy to grape. However, this is generally not the case in the financial area. To finish this project, I need to combine the data from 4 data sources (which use different identifiers of firms):
- Thomason Reuter's SDC Platinum
    - provide comprehensive MA event data
- Compustat (financial variable)
    - Unfortunately, there are a lot missing values in this dataset(poor quality). I suggest big data guy should not consider using this database.
- EDGAR (financial disclosures)
    - operated by SEC. Since the annual disclosure (10-K) is required by law. There is no missing data. However, they are all raw textutal data.
- TNIC (Text-based Network Industry Classifications Data)
    - TNIC is created based on bag-of-words (may seems to be a out-of-date NLP technique in Machine Learning area.). However, TNIC is surprisingly popular in Finance area.


In case of modeling, I use a Deep Learning Approach to learn the conditional intensity function(CIF) of temporal point process. There are several benefits to use point process to model the MA event sequence of firms.
- CIF can be deep-learning parameterized
- the sparsity is modeled inside the CIF (so no resampling techniques required)
- can make continuous prediction (compared with logistic regression, where the prediction period is ad-hoc)

The overall model structure:
<img src="https://github.com/dayuyang1999/MA_packed/blob/master/structure.png" alt="png" width="500" height="300"/>

How I combine Temporal Point Process(TPP) and Graph Neural Network (GNN):


<img src="https://github.com/dayuyang1999/MA_packed/blob/master/tpp-gnn.png" alt="png" width="500" height="200"/>




---
# files

Since the data is very large (over 50 GB), it cannot be uploaded to github, the whole process of data processing is in `MA` directory

For the rest of files:
- model.py: the model design
- data_preprocess.py: design the pytorch data_loader object; each time, load the timeline of a firm
- main.py: training process
- wandb: training caches



---



## Training Cache:

![](https://cdn.mathpix.com/snip/images/IT6TxGhQT9dbZjif3qv-tc_cpVT_ZNJZoHdqlGMN-Kc.original.fullsize.png)

