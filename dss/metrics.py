# Copyright 2023 The Distilling-step-by-step authors

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import numpy as np
import re
import string
from seqeval.metrics import classification_report, f1_score
from seqeval_custom import classification_report as cr_custom
from nltk import word_tokenize

def compute_text_acc(preds, labels):
    return np.mean(np.array(preds) == np.array(labels))


def compute_equation_acc(preds, labels):
    preds = [eval_equation(pred) for pred in preds]
    labels = [eval_equation(label) for label in labels]

    return np.mean(np.array(preds) == np.array(labels))


def eval_equation(equation):
    try:
        answer = eval(equation)
    except:
        answer = np.nan

    return answer

def compute_metrics_ner(tokenizer, label_list, result_file):
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        decoded_preds = np.where(predictions[0] != -100, predictions[0], tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(decoded_preds, skip_special_tokens=True)
        
        decoded_inputs = np.where(predictions[-1] != -100, predictions[-1], tokenizer.pad_token_id)
        decoded_inputs = tokenizer.batch_decode(decoded_inputs, skip_special_tokens=True)
        
        labels = np.where(labels[0] != -100, labels[0], tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        y_pred = [process_outputs(word_tokenize(i), p, label_list) for i,p in zip(decoded_inputs,decoded_preds)]
        y_true = [process_outputs(word_tokenize(i), l, label_list) for i,l in zip(decoded_inputs,decoded_labels)]
        
        ner_report_exact = classification_report(y_true, y_pred, digits=4, zero_division=0)
        micro_f_exact = f1_score(y_true, y_pred, average='micro')
        macro_f_exact = f1_score(y_true, y_pred, average='macro')
        ner_report_relaxed, micro_f_relaxed, macro_f_relaxed = cr_custom(y_true, y_pred, digits=4, criteria='relaxed')
        
        print(ner_report_exact)
        print(ner_report_relaxed)
        result_string = ner_report_exact+'\n'+ner_report_relaxed+'\n'
        write_file(result_string, result_file)
        return {'micro-f':micro_f_exact, 'macro-f': macro_f_exact,'micro-f_relaxed':micro_f_relaxed, 'macro-f_relaxed': macro_f_relaxed}
        
    return compute_metrics

def compute_metrics_text(tokenizer):
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions[0], skip_special_tokens=True)

        labels = np.where(labels[0] != -100, labels[0], tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        acc = np.mean(np.array(decoded_preds) == np.array(decoded_labels))

        return {'accuracy': acc}

    return compute_metrics


def compute_metrics_text_aux(tokenizer):
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        acc = np.mean(np.array(decoded_preds) == np.array(decoded_labels))

        return {'accuracy': acc}

    return compute_metrics



def compute_metrics_equation(tokenizer):
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions[0], skip_special_tokens=True)

        labels = np.where(labels[0] != -100, labels[0], tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        preds = list()
        for pred in decoded_preds:    
            preds.append(eval_equation(pred))

        labels = list()
        for label in decoded_labels:    
            labels.append(eval_equation(label))

        acc = np.mean(np.array(preds) == np.array(labels))

        return {'accuracy': acc}
    
    return compute_metrics


def compute_metrics_equation_aux(tokenizer):
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        preds = list()
        for pred in decoded_preds:    
            preds.append(eval_equation(pred))

        labels = list()
        for label in decoded_labels:    
            labels.append(eval_equation(label))

        acc = np.mean(np.array(preds) == np.array(labels))

        return {'accuracy': acc}
    
    return compute_metrics

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

            # generated label word might be noisy or lack proper space with the next word
            # so match the first few characters to see if it is a valid label
            if label not in label_list:
                match_flag = False
                for l in label_list:
                    if label.startswith(l): 
                        label = l 
                        match_flag = True
                        break
                if not match_flag: label = 'O' # if no match, set label to 'O'

            if token not in track_tokens: 
                track_tokens += [token]
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

def fmt_str(s):
    s = s.replace("'s",'').translate(str.maketrans('', '', string.punctuation)) 
    return s

def write_file(input,filename,mode='a'):
    '''
    Write `input` to the `filename`.
    '''
    with open(filename,mode) as f:
        f.write(str(input))