import math
import os
import re
import random
import numpy as np
import pandas as pd
import torch
import json
import string

from seqeval_custom import classification_report as cr_custom
from tqdm.auto import tqdm

def fmt_str(s):
    s = s.replace("'s",'').translate(str.maketrans('', '', string.punctuation)) 
    return s

def process_outputs(input, gen_label, label_list, constrained_decoding=False):
    '''
    Process the generated outputs and convert them into BIO tag scheme
    for evaluation.
    Expected format for `gen_label`: WORD1<TAG1>, WORD2<TAG2>, ... or WORD1 <TAG1>, WORD2 <TAG2>, ..
    '''
    final_labels = []
    track_tokens = []
    tokens, labels, phrase_len = [], [], []
    candidte_text_tag_pairs = gen_label.split('> ') if constrained_decoding else gen_label.split('>, ')
    for l in candidte_text_tag_pairs: # separate all predicted entities
        # try-catch block as some generation might have no ner tag or noisy output without any pattern
        try:
            if ' <' in l: 
                token, label = l.split(' <') # separate entity and its label
            else:
                token, label = l.split('<') # separate entity and its label
            # label = label[:-1] # remove tailing '>'
            if token in track_tokens: continue
            
            # generated label word might be noisy or lack proper space with the next word
            # so match the first few characters to see if it is a valid label
            # print(label)
            if label not in label_list:
                match_flag = False
                for l in label_list:
                    if label.startswith(l): 
                        label = l 
                        match_flag = True
                        break
                if not match_flag: label = 'O' # if no match, set label to 'O'
            
            # track_tokens += [token]
            token = re.split(" |'",token) # it may have multiple words
            label = ['B-'+label]+['I-'+label]*(len(token)-1) # set 'B-' and 'I-' prefixed accordingly
            tokens += token
            labels += label
            phrase_len += [len(tokens)-len(phrase_len)]*(len(tokens)-len(phrase_len))
        except:
            pass
        
    tag_ptr = 0 # index for 'tokens' and 'labels' 
    for idx,t in enumerate(input): # iterate over the input sentence, token by token
        if tokens:
            # Remove apostrophes and punctuations
            gold_token = fmt_str(t)
            pred_token = fmt_str(tokens[tag_ptr])
        if tokens and gold_token==pred_token: # set label to a token when the token matches with the generated token
            if labels[tag_ptr].startswith('B-') and tag_ptr<len(tokens)-1 and idx+1<len(input) and labels[tag_ptr+1]=='I-'+labels[tag_ptr][2:]:
                cmplt_gold_seq = ' '.join([fmt_str(i) for i in input])
                cmplt_gold_phrase = ' '.join([fmt_str(input[idx+i]) for i in range(phrase_len[tag_ptr]) if idx+i<len(input)])
                cmplt_pred_phrase = ' '.join([fmt_str(tokens[tag_ptr+i]) for i in range(phrase_len[tag_ptr])])
                nxt_gold_token = fmt_str(input[idx+1])
                nxt_pred_token = fmt_str(tokens[tag_ptr+1])
                if nxt_gold_token != nxt_pred_token:
                    final_labels += ['O']
                    continue
                if nxt_gold_token == nxt_pred_token and cmplt_pred_phrase not in cmplt_gold_phrase and cmplt_pred_phrase in cmplt_gold_seq:
                    final_labels += ['O']
                    continue
            final_labels += [labels[tag_ptr]]
            tag_ptr += 1
            if tag_ptr == len(tokens): break
        else: # otherwise set the token label to 'O'
            final_labels += ['O']
    final_labels += ['O']*(len(input)-len(final_labels))

    return final_labels

def process_outputs_text_infilling(input, gen_label, label_to_sentinel_token, is_v8=False):
    '''
    Process the generated outputs and convert them into BIO tag scheme
    for evaluation.
    Example >>
    Example for `gen_label`
        flan-T5:  <extra_id_1> European Commission, <extra_id_3> German, <extra_id_3> British</s>
        T5: <extra_id_1> European Commission,<extra_id_3> German,<extra_id_3> British</s>
        Or, when is_v8=True
        flan-T5:  <extra_id_1> European Commission, <extra_id_3> German <token_sep> British</s>
        T5: <extra_id_1> European Commission,<extra_id_3> German <token_sep> British</s>
    '''
    final_labels = []
    tokens, labels, phrase_len = [], [], []
    sentinel_to_label_token = {j:i for i,j in label_to_sentinel_token.items()}
    gen_label = gen_label.replace('</s>','') # remove tailing </s> and <pad> tokens
    gen_label = gen_label.replace('<pad>','') # remove <pad> token at the start
    gen_label = gen_label.replace('\n','') # remove \n token at the start
    candidte_text_tag_pairs = gen_label.split(', <') if ', <' in gen_label else gen_label.split(',<')
    for l in candidte_text_tag_pairs: # separate all predicted entities
        # try-catch block as some generation might have no ner tag or noisy output without any pattern
        try:
            l = l.replace('<token_sep>','[token_sep]')
            if '> ' in l: 
                label, token = l.split('> ') # separate entity and its label
            else:
                label, token = l.split('>') # separate entity and its label

            if label[-1]=='>':label = label[:-1] # remove tailing '>'
            if label[0]=='<':label=label[1:] # remove '<' at the start
            label = '<'+label+'>' 
            
            # generated label word might be noisy or lack proper space with the next word
            # so match the first few characters to see if it is a valid label
            if label not in sentinel_to_label_token:
                match_flag = False
                for l in sentinel_to_label_token:
                    if label.startswith(l): 
                        label = sentinel_to_label_token[l][1:-1] 
                        match_flag = True
                        break
                if not match_flag: label = 'O' # if no match, set label to 'O'
            else:
                label = sentinel_to_label_token[label][1:-1] # map to originl label from sentinel token

            consec_tokens = token.split(' [token_sep] ') if is_v8 else [token]
            for token in consec_tokens:
                token = re.split(" |'",token) # it may have multiple words
                cur_label = ['B-'+label]+['I-'+label]*(len(token)-1) # set 'B-' and 'I-' prefixed accordingly
                tokens += token
                labels += cur_label
            phrase_len += [len(tokens)-len(phrase_len)]*(len(tokens)-len(phrase_len))
        except:
            pass
    
    if not is_v8:
        tag_ptr = 0 # index for 'tokens' and 'labels' 
        for idx,t in enumerate(input): # iterate over the input sentence, token by token
            if tokens:
                # Remove apostrophes and punctuation
                gold_token = fmt_str(t)
                pred_token = fmt_str(tokens[tag_ptr])

            if tokens and gold_token==pred_token: # set label to a token when the token matches with the generated token
                if labels[tag_ptr].startswith('B-') and tag_ptr<len(tokens)-1 and idx+1<len(input) and labels[tag_ptr+1]=='I-'+labels[tag_ptr][2:]:
                    cmplt_gold_seq = ' '.join([fmt_str(i) for i in input])
                    cmplt_gold_phrase = ' '.join([fmt_str(input[idx+i]) for i in range(phrase_len[tag_ptr]) if idx+i<len(input)])
                    cmplt_pred_phrase = ' '.join([fmt_str(tokens[tag_ptr+i]) for i in range(phrase_len[tag_ptr])])
                    nxt_gold_token = fmt_str(input[idx+1])
                    nxt_pred_token = fmt_str(tokens[tag_ptr+1])
                    if nxt_gold_token != nxt_pred_token:
                        final_labels += ['O']
                        continue
                    if nxt_gold_token == nxt_pred_token and cmplt_pred_phrase not in cmplt_gold_phrase and cmplt_pred_phrase in cmplt_gold_seq:
                        final_labels += ['O']
                        continue
                final_labels += [labels[tag_ptr]]
                tag_ptr += 1
                if tag_ptr == len(tokens): break
            else: # otherwise set the token label to 'O'
                final_labels += ['O']
        final_labels += ['O']*(len(input)-len(final_labels))
    else:
        final_labels = ['O']*len(input)
        for index,t in enumerate(input): # iterate over the input sentence, token by token
            if tokens:
                for tok,label in zip(tokens,labels):
                    # Remove apostrophes and punctuations
                    gold_token = fmt_str(t)
                    pred_token = fmt_str(tok)
                    if gold_token==pred_token: # set label to a token when the token matches with the generated token
                        final_labels[index] = label

    return final_labels


label_to_sentinel_token_dictlist = {
    'conll2003' : {'<PER>': '<extra_id_0>', '<ORG>': '<extra_id_1>', '<LOC>': '<extra_id_2>', '<MISC>': '<extra_id_3>'},
    'conll2003_custom' : {'<PER>': '<sentinel_token_0>', '<ORG>': '<sentinel_token_1>', '<LOC>': '<sentinel_token_2>', '<MISC>': '<sentinel_token_3>'},
    'mit_movie_custom' : {'<Actor>': '<sentinel_token_0>', '<Plot>': '<sentinel_token_1>', '<Opinion>': '<sentinel_token_2>', 
                          '<Award>': '<sentinel_token_3>', '<Year>': '<sentinel_token_4>', '<Genre>': '<sentinel_token_5>', 
                          '<Origin>': '<sentinel_token_6>', '<Director>': '<sentinel_token_7>', '<Soundtrack>': '<sentinel_token_8>', 
                          '<Relationship>': '<sentinel_token_9>', '<Character_Name>': '<sentinel_token_10>', '<Quote>': '<sentinel_token_11>'},
    'mit_movie' : {'<Actor>': '<extra_id_0>', '<Plot>': '<extra_id_1>', '<Opinion>': '<extra_id_2>', '<Award>': '<extra_id_3>', '<Year>': '<extra_id_4>', 
                   '<Genre>': '<extra_id_5>', '<Origin>': '<extra_id_6>', '<Director>': '<extra_id_7>', '<Soundtrack>': '<extra_id_8>', '<Relationship>': '<extra_id_9>', 
                   '<Character_Name>': '<extra_id_10>', '<Quote>': '<extra_id_11>'},
    'bleeding' : {'<BLEEDING_ANATOMIC_SITE>': '<extra_id_0>', '<BLEEDING_EVENT>': '<extra_id_1>', '<BLEEDING_LAB_EVAL>': '<extra_id_2>', 
                  '<DRUGNAME>': '<extra_id_3>', '<SEVERITY>': '<extra_id_4>', '<TRIGGER_ALTERNATIVE_CAUSE>': '<extra_id_5>'},
    'bleeding_custom' : {'<BLEEDING_ANATOMIC_SITE>': '<sentinel_token_0>', '<BLEEDING_EVENT>': '<sentinel_token_1>', '<BLEEDING_LAB_EVAL>': '<sentinel_token_2>', 
                         '<DRUGNAME>': '<sentinel_token_3>', '<SEVERITY>': '<sentinel_token_4>', '<TRIGGER_ALTERNATIVE_CAUSE>': '<sentinel_token_5>'},
    'mit_restaurant' : {'<Rating>': '<extra_id_0>', '<Amenity>': '<extra_id_1>', '<Location>': '<extra_id_2>', '<Restaurant_Name>': '<extra_id_3>', 
                        '<Price>': '<extra_id_4>', '<Hours>': '<extra_id_5>', '<Dish>': '<extra_id_6>', '<Cuisine>': '<extra_id_7>'},
    'mit_restaurant_custom' : {'<Rating>': '<sentinel_token_0>', '<Amenity>': '<sentinel_token_1>', '<Location>': '<sentinel_token_2>', 
                               '<Restaurant_Name>': '<sentinel_token_3>', '<Price>': '<sentinel_token_4>', '<Hours>': '<sentinel_token_5>', 
                               '<Dish>': '<sentinel_token_6>', '<Cuisine>': '<sentinel_token_7>'},
    'sbdh_gpt4_v2':{'<barriers_to_care>': '<extra_id_0>', '<financial_insecurity>': '<extra_id_1>', '<food_insecurity>': '<extra_id_2>', 
                 '<housing_insecurity>': '<extra_id_3>', '<isolation_or_loss_of_relationship>': '<extra_id_4>', '<legal_problems>': '<extra_id_5>', 
                 '<pain>': '<extra_id_6>', '<patient_disability>': '<extra_id_7>', '<psychiatric_symptoms_or_disorders>': '<extra_id_8>', 
                 '<substance_abuse>': '<extra_id_9>', '<transitions_of_care>': '<extra_id_10>', '<violence>': '<extra_id_11>'},
    'sbdh_va':{'<barriers_to_care>': '<extra_id_0>', '<financial_insecurity>': '<extra_id_1>', '<food_insecurity>': '<extra_id_2>', 
               '<housing_insecurity>': '<extra_id_3>', '<isolation_or_loss_of_relationship>': '<extra_id_4>', '<legal_problems>': '<extra_id_5>', 
               '<pain>': '<extra_id_6>', '<patient_disability>': '<extra_id_7>', '<psychiatric_symptoms_or_disorders>': '<extra_id_8>', 
               '<substance_abuse>': '<extra_id_9>', '<transitions_of_care>': '<extra_id_10>', '<violence>': '<extra_id_11>'},
}

if __name__ == '__main__':
    pred_files = [
        # 'flan_t5_base_preds_sbdh_gpt4_msf_v3_0_noPtr.txt',
        # 'flan_t5_base_preds_sbdh_gpt4_msf_v3_1_noPtr.txt',
        # 'flan_t5_base_preds_sbdh_gpt4_msf_v3_2_noPtr.txt',
        # 't5v1_1_base_preds_sbdh_gpt4_msf_v3_0_noPtr.txt',
        # 't5v1_1_base_preds_sbdh_gpt4_msf_v3_1_noPtr.txt',
        # 't5v1_1_base_preds_sbdh_gpt4_msf_v3_2_noPtr.txt',
        # 'flan_t5_base_preds_conll2003_.4_0_textInfill_v3.txt',
        # 'flan_t5_base_preds_conll2003_.4_1_textInfill_v3.txt',
        # 'flan_t5_base_preds_conll2003_.4_2_textInfill_v3.txt',
        # 'flan_t5_base_preds_conll2003_.2_0_textInfill_v3.txt',
        # 'flan_t5_base_preds_conll2003_.2_1_textInfill_v3.txt',
        # 'flan_t5_base_preds_conll2003_.2_2_textInfill_v3.txt',
        # 'flan_t5_base_preds_conll2003_.1_0_textInfill_v3.txt',
        # 'flan_t5_base_preds_conll2003_.1_1_textInfill_v3.txt',
        # 'flan_t5_base_preds_conll2003_.1_2_textInfill_v3.txt',
        # 'flan_t5_base_preds_conll2003_.05_0_textInfill_v3.txt',
        # 'flan_t5_base_preds_conll2003_.05_1_textInfill_v3.txt',
        # 'flan_t5_base_preds_conll2003_.05_2_textInfill_v3.txt',
        # 'flan_t5_base_preds_conll2003_.01_0_textInfill_v3.txt',
        # 'flan_t5_base_preds_conll2003_.01_1_textInfill_v3.txt',
        # 'flan_t5_base_preds_conll2003_.01_2_textInfill_v3.txt',
        # 'flan_t5_base_preds_conll2003_0.005_0_textInfill_v3.txt',
        # 'flan_t5_base_preds_conll2003_0.005_1_textInfill_v3.txt',
        # 'flan_t5_base_preds_conll2003_0.005_2_textInfill_v3.txt',
        # 'flan_t5_base_preds_conll2003_.4_0_textInfill_v5.txt',
        # 'flan_t5_base_preds_conll2003_.4_1_textInfill_v5.txt',
        # 'flan_t5_base_preds_conll2003_.4_2_textInfill_v5.txt',
        # 'flan_t5_base_preds_conll2003_.2_0_textInfill_v5.txt',
        # 'flan_t5_base_preds_conll2003_.2_1_textInfill_v5.txt',
        # 'flan_t5_base_preds_conll2003_.2_2_textInfill_v5.txt',
        # 'flan_t5_base_preds_conll2003_.1_0_textInfill_v5.txt',
        # 'flan_t5_base_preds_conll2003_.1_1_textInfill_v5.txt',
        # 'flan_t5_base_preds_conll2003_.1_2_textInfill_v5.txt',
        # 'flan_t5_base_preds_conll2003_.05_0_textInfill_v5.txt',
        # 'flan_t5_base_preds_conll2003_.05_1_textInfill_v5.txt',
        # 'flan_t5_base_preds_conll2003_.05_2_textInfill_v5.txt',
        # 'flan_t5_base_preds_conll2003_.01_0_textInfill_v5.txt',
        # 'flan_t5_base_preds_conll2003_.01_1_textInfill_v5.txt',
        # 'flan_t5_base_preds_conll2003_.01_2_textInfill_v5.txt',
        # 'flan_t5_base_preds_conll2003_0.005_0_textInfill_v5.txt',
        # 'flan_t5_base_preds_conll2003_0.005_1_textInfill_v5.txt',
        # 'flan_t5_base_preds_conll2003_0.005_2_textInfill_v5.txt',
    ]
    for pred_file in tqdm(pred_files):
        print(pred_file)
        all_inputs,all_labels,all_preds = [],[],[]
        y_true, y_pred = [], []
        
        with open(os.path.join('/home/avijit/playground/generative_ner/output',pred_file)) as f:
            dataset = None
            if 'bleeding' in pred_file:
                dataset = 'bleeding'
            elif 'mit_movie' in pred_file:
                dataset = 'mit_movie'
            elif 'mit_restaurant' in pred_file:
                dataset = 'mit_restaurant'
            elif 'sbdh_gpt4_v2' in pred_file:
                dataset = 'sbdh_gpt4_v2'
            elif 'conll2003' in pred_file:
                dataset = 'conll2003'
            else:
                dataset = 'sbdh_va'
                
            is_custom_sentinel_token = False
            for i,line in enumerate(f.readlines()):
                line = line.rstrip()
                if i%3==0:all_inputs += [line[6:].split('</s>')[0]]
                if i%3==1:
                    all_labels += [line[6:]]
                    if not is_custom_sentinel_token and '<sentinel_token' in line[6:]: 
                        is_custom_sentinel_token = True
                
                if i%3==2:
                    all_preds += [line[6:]]

        if is_custom_sentinel_token:
            label_to_sentinel_token = label_to_sentinel_token_dictlist[dataset+'_custom']
        else:
            label_to_sentinel_token = label_to_sentinel_token_dictlist[dataset]
        for i,g,p in zip(all_inputs,all_labels,all_preds):
            input_tokens = re.split(" |'",i)
            if '_textInfill_' in pred_file:
                y_true += [process_outputs_text_infilling(input_tokens, g, label_to_sentinel_token, is_v8=False)]
                y_pred += [process_outputs_text_infilling(input_tokens, p, label_to_sentinel_token, is_v8=False)]
            else:
                label_list = [label[1:-1] for label in label_to_sentinel_token_dictlist[dataset].keys()]
                y_true += [process_outputs(input_tokens, g, label_list)]
                y_pred += [process_outputs(input_tokens, p, label_list)]

        exact_report = cr_custom(y_true, y_pred, digits=4, criteria='exact')[0]
        relaxed_report = cr_custom(y_true, y_pred, digits=4, criteria='relaxed')[0]

        with open(os.path.join('/home/avijit/playground/generative_ner/best_result',pred_file.replace('_preds_','_best_result_').replace('.txt','_relaxed.txt')),'w') as f:
            f.write(exact_report+'\n')
            f.write(relaxed_report)