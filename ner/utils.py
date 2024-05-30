import nltk
import torch
import json
import string
import re
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from transformers.models.bart.modeling_bart import shift_tokens_right
from dataclasses import dataclass
from typing import Optional, Union
from collections import Counter
from transformers.file_utils import ModelOutput
from transformers import DataCollatorForTokenClassification
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy

def postprocess_text(preds, labels):
    '''
    Psot process text for generation metrics (ROUGE/Bleu).
    '''
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]
    
    # BLEU expects tokenized sequences
    tok_preds = [nltk.word_tokenize(pred) for pred in preds] 
    tok_labels = [[nltk.word_tokenize(label)] for label in labels]

    # rougeLSum expects newline after each sentence
    r_preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    r_labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels, tok_preds, tok_labels, r_preds, r_labels

def get_labels(y_pred, y_true, label_list):
    '''
    Get token labels for the NER task.
    '''
    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(pred, gold_label) if l != -100]
        for pred, gold_label in zip(y_pred, y_true)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(pred, gold_label) if l != -100]
        for pred, gold_label in zip(y_pred, y_true)
    ]
    return true_predictions, true_labels

def write_preds(inputs, predictions, references, filename):
    '''
    Write gold (references) and model (predictions) rewrites in a text file (filename).
    '''
    with open(filename,'w') as f:
        for i,j,k in zip(inputs, predictions, references):
            f.write('Inpt: '+i.replace('\n',' ')+'\n')
            f.write('Gold: '+k.replace('\n',' ')+'\n')
            f.write('Pred: '+j.replace('\n',' ')+'\n')

def write_file(input,filename,mode='a'):
    '''
    Write `input` to the `filename`.
    '''
    with open(filename,mode) as f:
        f.write(str(input))

def get_weight(file_name,aux_task,label2id,use_sklearn=True):
    '''
    Get weight for cost-sensitive learning.
    '''
    with open(file_name) as f:
        data = json.load(f)
    if aux_task == 'hal':
        labels = [label2id[i[aux_task]] for i in data['data']]
    elif aux_task == 'domain':
        labels = [label2id.get(i[aux_task],label2id['Others']) for i in data['data']]
    
    if not use_sklearn: # Use log weight
        alpha = 15 # a hyper-parameter
        classes = np.unique(labels)
        class_weights = []
        class_count_dict = dict(Counter(labels))
        for i in classes:
            weight = np.log10(alpha*len(labels)/class_count_dict[i])
            if weight < 1: weight = 1.
            class_weights += [weight]
        class_weights = np.array(class_weights)
    else: # Use sklearn's default method
        class_weights = compute_class_weight(
            class_weight = 'balanced',
            classes = np.array(list(label2id.values())), 
            y = labels)
    return class_weights

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

class DataCollatorForEncoderDecoderModel:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.
    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        model (:class:`~transformers.PreTrainedModel`):
            The model that is being trained. If set and has the `prepare_decoder_input_ids_from_labels`, use it to
            prepare the `decoder_input_ids`
            This is useful when using `label_smoothing` to avoid calculating loss twice.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (:obj:`int`, `optional`, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
    """

    def __init__(self, tokenizer, model, pad_token_id, decoder_start_token_id, padding = True, max_length = None, pad_to_multiple_of = None, label_pad_token_id = -100):
        
        self.tokenizer = tokenizer
        self.model = model
        self.padding = padding
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.label_pad_token_id = label_pad_token_id
        self.pad_token_id = pad_token_id
        self.decoder_start_token_id = decoder_start_token_id

    def __call__(self, features):
        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                feature["labels"] = (
                    feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                )

        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # prepare decoder_input_ids
        decoder_input_ids = shift_tokens_right(features["labels"], self.pad_token_id, self.decoder_start_token_id)
        features["decoder_input_ids"] = decoder_input_ids

        return features

@dataclass
class DataCollatorForSeq2SeqAndNer:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.
    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        model (:class:`~transformers.PreTrainedModel`):
            The model that is being trained. If set and has the `prepare_decoder_input_ids_from_labels`, use it to
            prepare the `decoder_input_ids`
            This is useful when using `label_smoothing` to avoid calculating loss twice.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (:obj:`int`, `optional`, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
    """

    def __init__(self, tokenizer, model, padding = True, max_length = None, pad_to_multiple_of = None, label_pad_token_id = -100):
        
        self.tokenizer = tokenizer
        self.model = model
        self.padding = padding
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.label_pad_token_id = label_pad_token_id

    def __call__(self, features):
        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        ner_labels = [feature["labels_aux_task"] for feature in features] if "labels_aux_task" in features[0].keys() else None
        
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                feature["labels"] = (
                    feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                )

        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt" if labels is None else None,
        )

        if ner_labels is not None:
            sequence_length = torch.tensor(features["input_ids"]).shape[1]
            padding_side = self.tokenizer.padding_side
            if padding_side == "right":
                features["labels_aux_task"] = [
                    label + [self.label_pad_token_id] * (sequence_length - len(label)) for label in ner_labels
                ]
            else:
                features["labels_aux_task"] = [
                    [self.label_pad_token_id] * (sequence_length - len(label)) + label for label in ner_labels
                ]

        features = {k: torch.tensor(v, dtype=torch.int64) for k, v in features.items()}

        # prepare decoder_input_ids
        if self.model is not None and hasattr(self.model, "prepare_decoder_input_ids_from_labels"):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
            features["decoder_input_ids"] = decoder_input_ids

        return features
    
class CustomDataCollatorForTokenClassification(DataCollatorForTokenClassification):
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single sequence
              is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
              lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (`int`, *optional*, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignore by PyTorch loss functions).
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def torch_call(self, features):
        import torch

        label_name = "labels"
        labels = [feature[label_name] for feature in features] if label_name in features[0].keys() else None
        presence_label_name = "presence_labels"
        presence_labels = [feature[presence_label_name] for feature in features] if presence_label_name in features[0].keys() else None
        period_label_name = "period_labels"
        period_labels = [feature[period_label_name] for feature in features] if period_label_name in features[0].keys() else None
        
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            # Conversion to tensors will fail if we have labels as they are not of the same length yet.
            return_tensors="pt" if labels is None else None,
        )

        if labels is None:
            return batch

        sequence_length = torch.tensor(batch["input_ids"]).shape[1]
        padding_side = self.tokenizer.padding_side
        if padding_side == "right":
            batch[label_name] = [
                list(label) + [self.label_pad_token_id] * (sequence_length - len(label)) for label in labels
            ]
            batch[presence_label_name] = [
                list(label) + [self.label_pad_token_id] * (sequence_length - len(label)) for label in presence_labels
            ]
            batch[period_label_name] = [
                list(label) + [self.label_pad_token_id] * (sequence_length - len(label)) for label in period_labels
            ]
        else:
            batch[label_name] = [
                [self.label_pad_token_id] * (sequence_length - len(label)) + list(label) for label in labels
            ]
            batch[presence_label_name] = [
                [self.label_pad_token_id] * (sequence_length - len(label)) + list(label) for label in presence_labels
            ]
            batch[period_label_name] = [
                [self.label_pad_token_id] * (sequence_length - len(label)) + list(label) for label in period_labels
            ]

        batch = {k: torch.tensor(v, dtype=torch.int64) for k, v in batch.items()}
        return batch