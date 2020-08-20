# Import the necessary packages
import pandas as pd
import pickle
from datetime import datetime
# import argparse
import argparse

import sys
sys.path.insert(1, '../common')
# Utlity functions from us..
import nlp_utils as nu
from models import run_all_pipelines

def run_pipeline(data):
    X = data['text']
    y = data['target']

    print("Launching pipeline")
    results = run_all_pipelines(X,y,grid_search=True)
    #results = run_all_pipelines(X,y)

    print("Results:", results)
    res_df = pd.DataFrame(results)
    res_df = res_df.sort_values(by='recall', ascending=False)
    now = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    res_df.to_csv('results'+now+'.csv',  encoding='utf-8', index=False)    


# run with short data
def dry_run():
    """
    Simple dry run to check the pipeline before kicking off full dataset
    """
    print("Loading data file")
    data = pickle.load(open('../../Gaussian Solutions Input Data/sanity_df.pkl', 'rb'))
    run_pipeline(data)

# run with short data
def full_run():
    """
    Simple dry run to check the pipeline before kicking off full dataset
    """
    print("Loading data file")
    data = pickle.load(open('../../Gaussian Solutions Input Data/gsa_train.pkl', 'rb'))
    run_pipeline(data)

# run with short data
def aug_run():
    """
    Simple dry run to check the pipeline before kicking off full dataset
    """
    print("Loading data file")
    data = pickle.load(open('./aug_tl_df_all.pkl', 'rb'))
    #data = pickle.load(open('./aug_tl_df_truc_uw_500.pkl', 'rb'))
    run_pipeline(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run NLP Pipeline')
    parser.add_argument('--dry_run', help="True: runs with short data set, \n False:  runs with the full data set",
        type=int, required=True)
    args = parser.parse_args()
    print(vars(args))
    
    if args.dry_run == 1:
        print("Running a dry run")
        dry_run()
    else:
        print("full data set run")
        full_run()
        #aug_run()

    
