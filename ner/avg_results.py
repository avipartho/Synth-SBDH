import argparse, os
import numpy as np
import pandas as pd
import itertools
'''
Run 
    `python avg_results.py --w_relaxed_matching`
'''
def avg_results(result_dict,args,total_runs=3):
    result = {}
    for k in result_dict[0]:
        np_metric = np.array([result_dict[seed][k] for seed in range(total_runs)])
        mean, std = np_metric.mean(axis=0), np_metric.std(axis=0)
        # maen_std = ['{:.3f} ± {:.3f} %'.format(i,j*100) for i,j in zip(mean,std)]
        mean_std = ['{:.2f} ± {:.2f}'.format(i*100,j*100) for i,j in zip(mean,std)]
        result[k] = list(mean_std)
    
    df = pd.DataFrame(result).T.rename({0:'precision',1:'recall',2:'f-score'},axis=1)
    if args.verbose: print(df)
    elif args.micro_macro_only: print(df.loc[['micro','macro'],:])
    return df

def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--num_labels",type=int,required=True,help='Number of labels for the NER dataset.')
    parser.add_argument("--verbose",action='store_true',help='Set this flag if you want to see detailed results.')
    parser.add_argument("--micro_macro_only",action='store_true',help='Set this flag if you only want to see micro and macro metrics.')
    parser.add_argument("--w_relaxed_matching",action='store_true',help='Set this flag if you want to see macro metrics with both exact and relaxed matching (only for test set).')
    args = parser.parse_args()

    # interested experiments
    exp_suffix_list = [
        '_decOnly_1e5', '_encOnly_1e4', '_encOnly_1e5', '_encOnly_5e5', '_encOnly_9e6', 
        '_noPtr', '_ptr', '_ptr_v1', '_ptr_v2', '_ptr_v3', '_ptr_v4', '_ptr_v5', 
        '_textInfill_v1', '_textInfill_v2', '_textInfill_v3', '_textInfill_v3_1e4', '_textInfill_v3_1e5', '_textInfill_v3_2', 
        '_textInfill_v4', '_textInfill_v5', '_textInfill_v6', '_textInfill_v7', '_textInfill_v10', '_textInfill_v11', 
        '_textInfill_v12', '_textInfill_v13', '_textInfill_v14', '_textInfill_v15', '_textInfill_v16', '_textInfill_v17',
        ''        
        ]
    model_list = [
        't5v1_1_base', 't5_base', 'flan_t5_base', 'flan_t5_base_pretrained', 
        'clinical_t5_base', 'clinical_t5_sci',
        'gpt2', 'roberta_base', 'cliroberta_base',
        'llama3_2_3b', 'llama3_8b'
        ]
    exp_model_lists = list(itertools.product(model_list, exp_suffix_list)) 
    # + \
    #     list(itertools.product(model_list[4:5],exp_suffix_list[1:5])) + [('gpt2', '_decOnly_1e5')]
    # [
    #     # ('_constDec_v2','flan_t5_base'),
    #     # ('_constDec_v3','flan_t5_base'),
    #     # ('_sprompt_1e4_v1','roberta_base'),
    #     # ('_sprompt_1e5_v1','roberta_base'),
    #     # ('_sprompt_5e5_v1','roberta_base'),
    #     # ('_sprompt_5e5_v2','roberta_base'),     
    #     # ('_sprompt_5e5_v3','roberta_base'),
    #     # ('_sprompt_5e5_v4','roberta_base'),
    #     # ('_sprompt_5e5_v5','roberta_base'),
    #     # ('_sprompt_5e5_v6','roberta_base'),
    #     # ('_sprompt_5e5_v7','roberta_base'),
    # ]
    
    dataset_label_dict = {
        'conll2003':4,
        'conll2003_.005':4,
        'conll2003_.01':4,
        'conll2003_.05':4,
        'conll2003_.1':4,
        'conll2003_.2':4,
        'conll2003_.4':4,
        'mit_restaurant':8,
        'mit_movie':12,
        'bleeding':6,
        'sbdh_gpt4':12,
        'sbdh_gpt4_v2':12,
        'sbdh_gpt4_msf':12,
        'sbdh_gpt4_msf_v3':12,
        }
       
    for dataset in dataset_label_dict:
        print(f'###### Results for {dataset} ######')
        all_result_df = pd.DataFrame(
            np.zeros((len(exp_model_lists),4)),
            index=[f'{k}{v}' for k,v in exp_model_lists],
            columns=['val-micro-f','val-macro-f','test-micro-f','test-macro-f'] if not args.w_relaxed_matching else ['micro-f (exact)','macro-f (exact)','micro-f (relaxed)','macro-f (relaxed)']
        )
        for model,exp_suffix in exp_model_lists:    
            file_name_suffix = '_relaxed' if args.w_relaxed_matching else ''
            indent = 2 if args.w_relaxed_matching else 0
                
            if not os.path.exists(f'{model}_best_result_{dataset}_0{exp_suffix}{file_name_suffix}.txt')\
                or not os.path.exists(f'{model}_best_result_{dataset}_1{exp_suffix}{file_name_suffix}.txt')\
                or not os.path.exists(f'{model}_best_result_{dataset}_2{exp_suffix}{file_name_suffix}.txt'): # if result file is missing for any seed, skip
                continue
            if args.verbose or args.micro_macro_only:print(f'###### Results for {exp_suffix} ######')
            # Read results
            total_runs = 3
            dev_results,test_results = {}, {}
            for seed in range(total_runs):
                num_labels = dataset_label_dict[dataset]
                with open(f'{model}_best_result_{dataset}_{seed}{exp_suffix}{file_name_suffix}.txt') as log:
                    lines = log.readlines()
                dev_log, test_log = {}, {}
                for l in lines[2:num_labels+7]:
                    words = l.split()
                    # print(words)
                    if not words:continue
                    if 'avg' in words: dev_log[words[0]] = np.array([float(w) for w in words[2:-1]])
                    else: dev_log[words[0]] = np.array([float(w) for w in words[1:-1]])
                for l in lines[num_labels+11-indent:]:
                    words = l.split()
                    if not words:continue
                    if 'avg' in words: test_log[words[0]] = np.array([float(w) for w in words[2:-1]])
                    else: test_log[words[0]] = np.array([float(w) for w in words[1:-1]])
                    
                dev_results[seed] = dev_log
                # print(test_log)
                test_results[seed] = test_log 
            
            # Average and print
            if args.verbose or args.micro_macro_only:print('=== Dev Set ===')
            dev_df = avg_results(dev_results,args,total_runs)
            if not args.w_relaxed_matching:
                all_result_df.loc[f'{model}{exp_suffix}','val-micro-f'] = dev_df.loc['micro','f-score']
                all_result_df.loc[f'{model}{exp_suffix}','val-macro-f'] = dev_df.loc['macro','f-score']
            else:
                all_result_df.loc[f'{model}{exp_suffix}','micro-f (exact)'] = dev_df.loc['micro','f-score']
                all_result_df.loc[f'{model}{exp_suffix}','macro-f (exact)'] = dev_df.loc['macro','f-score']
            if args.verbose or args.micro_macro_only:print('=== Test Set ===')
            test_df = avg_results(test_results,args,total_runs)
            if not args.w_relaxed_matching:
                all_result_df.loc[f'{model}{exp_suffix}','test-micro-f'] = test_df.loc['micro','f-score']
                all_result_df.loc[f'{model}{exp_suffix}','test-macro-f'] = test_df.loc['macro','f-score']
            else:
                all_result_df.loc[f'{model}{exp_suffix}','micro-f (relaxed)'] = test_df.loc['micro','f-score']
                all_result_df.loc[f'{model}{exp_suffix}','macro-f (relaxed)'] = test_df.loc['macro','f-score']
            
            # color = (all_test_df['macro-f'] == '0.9212 ± 0.0002').map({True: 'background-color: yellow', False: ''})
            # all_test_df.style.apply(lambda s: color)

        print()
        all_result_df = all_result_df.loc[~(all_result_df==0).all(axis=1)].copy()
        columns=['val-micro-f','val-macro-f','test-micro-f','test-macro-f'] if not args.w_relaxed_matching else ['micro-f (exact)','macro-f (exact)','micro-f (relaxed)','macro-f (relaxed)']
        # print(all_result_df)
        for metric in columns:
            metric_list = [float(i.split(' ± ')[0]) for i in all_result_df[metric].tolist()]
            # print('Best {:}: {:} for `{:}`'.format(metric, max(metric_list), all_result_df.index.values[metric_list.index(max(metric_list))]))
            all_result_df.iloc[metric_list.index(max(metric_list))][metric] = '↑ '+all_result_df.iloc[metric_list.index(max(metric_list))][metric]
            # '✔'
        print(all_result_df)
        all_result_df.to_csv(f'all_result_{dataset}.csv')
        print()

if __name__ == "__main__":
    main()
