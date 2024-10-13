import math
import torch
import re
import numpy as np
from tqdm.auto import tqdm
from utils import postprocess_text, write_preds, process_outputs, process_outputs_text_infilling, write_file
from seqeval.metrics import classification_report, f1_score
from seqeval_custom import classification_report as cr_custom

def process_target_inputs(inputs, target_input_ids, mapping, config):
    '''
    Prepare decoder_input_ids/target_ids so that it follows tokenizer's vocab ids
    inputs: Encoder input ids (from input sequence)
    target_input_ids: Generated sequence or gold label sequence to process
    mapping: A list of input_ids for special tokens and ner tags 
    config: Model config
    '''
    src_start_index = len(mapping)
    mapping = torch.LongTensor(mapping)
    target_input_ids.masked_fill_(target_input_ids == -100, mapping[config.pad_token_id])

    tag_token_mask = target_input_ids.lt(src_start_index)  
    tag_index = target_input_ids.masked_fill(target_input_ids.ge(src_start_index), 0) # Only keep tag & special token indices
    tag_tokens = mapping[tag_index].to(target_input_ids.device)

    src_tokens_index = target_input_ids - src_start_index 
    src_tokens_index = src_tokens_index.masked_fill(src_tokens_index.lt(0), 0) # Only keep src_token indices
    word_tokens = inputs.gather(index=src_tokens_index, dim=1)

    processed_ids = torch.where(tag_token_mask, tag_tokens, word_tokens)

    return processed_ids

def evaluate_e2dmodel(
    model, 
    config, 
    tokenizer, 
    eval_dataloader, 
    accelerator, 
    device, 
    logger, 
    n_gpu,  
    args, 
    metrics, 
    label_list, 
    label_to_sentinel_token=None,
    write_predictions=False,
    constrained_decoding=False, 
    mapping=None):

    if constrained_decoding and mapping is None:
        raise ValueError("For constrained decoding, set the mapping parameter")

    metric_rouge, metric_bleu = metrics
    model.eval()
    if args.val_max_target_length is None:
        args.val_max_target_length = args.max_target_length

    gen_kwargs = {
        "max_length": args.val_max_target_length if args is not None else config.max_length,
        "num_beams": args.num_beams,
    }

    all_inputs, all_preds, all_labels = [], [], []

    for step, batch in tqdm(enumerate(eval_dataloader),total=math.ceil(len(eval_dataloader)/args.gradient_accumulation_steps),
                            disable=not accelerator.is_local_main_process if args.use_accelerate else False):
        with torch.no_grad():
            if args.use_accelerate:
                generated_tokens = accelerator.unwrap_model(model).generate(
                    batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    output_attentions=True,
                    **gen_kwargs,
                )
                generated_tokens = accelerator.pad_across_processes(
                    generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
                )

                labels = batch["labels"]
                inputs = batch["input_ids"]
                if not args.pad_to_max_length: # If we did not pad to max length, we need to pad the labels too
                    labels = accelerator.pad_across_processes(batch["labels"], dim=1, pad_index=tokenizer.pad_token_id)
                    inputs = accelerator.pad_across_processes(batch["input_ids"], dim=1, pad_index=tokenizer.pad_token_id)
                
                inputs = accelerator.gather(inputs).cpu() 
                generated_tokens = accelerator.gather(generated_tokens).cpu()
                labels = accelerator.gather(labels).cpu()

                # print(generated_tokens[1])
                # print(labels[1])
                if constrained_decoding:
                    generated_tokens = process_target_inputs(inputs, generated_tokens, mapping, config).numpy()
                    labels = process_target_inputs(inputs, labels, mapping, config).numpy()
                else:
                    generated_tokens = generated_tokens.numpy()
                    labels = labels.numpy()
                inputs = inputs.numpy()
                # print(generated_tokens[1])
                # print(labels[1])
                # print(tokenizer.batch_decode(inputs[:2], skip_special_tokens=True))
                # print(tokenizer.batch_decode(labels[:2], skip_special_tokens=True))
                # print(tokenizer.batch_decode(generated_tokens[:2], skip_special_tokens=True))
                # exit()

            else:
                if args.no_cuda or n_gpu==1:
                    generated_tokens = model.generate(
                        batch["input_ids"].to(device),
                        attention_mask=batch["attention_mask"].to(device),
                        output_attentions=True,
                        **gen_kwargs,
                    )
                elif n_gpu>1:
                    generated_tokens = model.module.generate(
                        batch["input_ids"].to(device),
                        attention_mask=batch["attention_mask"].to(device),
                        **gen_kwargs,
                    )

                inputs = batch["input_ids"] 
                if constrained_decoding:
                    generated_tokens = process_target_inputs(inputs, generated_tokens.cpu(), mapping, config).numpy()
                    labels = process_target_inputs(inputs, batch["labels"], mapping, config).numpy()
                else:
                    generated_tokens = generated_tokens.cpu().numpy()     
                    labels = batch["labels"]
                inputs = inputs.numpy()

            if isinstance(generated_tokens, tuple):
                generated_tokens = generated_tokens[0]
            if args.ignore_pad_token_for_loss:
                # Replace -100 in the labels as we can't decode them.
                labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

            decoded_inputs = tokenizer.batch_decode(inputs, skip_special_tokens=True if label_to_sentinel_token is None else False)
            decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True if label_to_sentinel_token is None else False)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True if label_to_sentinel_token is None else False)
            
            decoded_preds, decoded_labels, decoded_tok_preds, decoded_tok_labels, decoded_r_preds, decoded_r_labels = postprocess_text(decoded_preds, decoded_labels)

            all_inputs += decoded_inputs
            all_preds += decoded_preds
            all_labels += decoded_labels

            # metric_rouge.add_batch(predictions=decoded_r_preds, references=decoded_r_labels)
            # metric_bleu.add_batch(predictions=decoded_tok_preds, references=decoded_tok_labels)

    if write_predictions: # save predictions 
        if not args.use_accelerate:
            write_preds(inputs=all_inputs, predictions=all_preds, references=all_labels, filename=args.pred_file)
        elif accelerator.is_local_main_process: 
            write_preds(inputs=all_inputs, predictions=all_preds, references=all_labels, filename=args.pred_file)

    
    # result_rouge = metric_rouge.compute(use_stemmer=True)
    # result_bleu = metric_bleu.compute()
    
    y_true, y_pred = [], []
    for i,g,p in zip(all_inputs,all_labels,all_preds):
        input_tokens = i.split('</s>')[0]
        input_tokens = re.split(" |'",input_tokens)
        if label_to_sentinel_token is None:
            y_true += [process_outputs(input_tokens, g, label_list, constrained_decoding=constrained_decoding)]
            y_pred += [process_outputs(input_tokens, p, label_list, constrained_decoding=constrained_decoding)]
        else:
            y_true += [process_outputs_text_infilling(input_tokens, g, label_to_sentinel_token, is_v8=True if args.prefix_scheme in ['v8','v9'] else False)]
            y_pred += [process_outputs_text_infilling(input_tokens, p, label_to_sentinel_token, is_v8=True if args.prefix_scheme==['v8','v9'] else False)]
            
            # print(input_tokens)
            # print(g)
            # print(p)
            # print(y_true[-1])
            # print(y_pred[-1])

    # for i in range(5):
    #     print(all_inputs[i])
    #     print(all_labels[i])
    #     print(all_preds[i])
    #     print(y_true[i])
    #     print(y_pred[i])
    # exit()
    
    ner_report = classification_report(y_true, y_pred, digits=4, zero_division=0)
    macro_f = f1_score(y_true, y_pred, average='macro')
    
    # Extract ROUGE scores, for precision/recall use value.mid.precision/value.mid.recall
    # result_rouge = {k: round(v.mid.fmeasure * 100, 4) for k, v in result_rouge.items()}

    # logger.info(result_rouge)
    # logger.info('BLEU: {:.4f}'.format(result_bleu["bleu"]*100))
    logger.info("\n%s", ner_report)

    # result_string = '\n'.join([str(result_rouge),'BLEU: {:.4f}'.format(result_bleu["bleu"]*100), ner_report])+'\n'
    result_string = ner_report+'\n'
    if not args.use_accelerate or accelerator.is_local_main_process:
        write_file(result_string,args.result_file)
    logger.info("***** Evaluation finished! *****")

    return macro_f, result_string

def evaluate_armodel(
    model, 
    config, 
    tokenizer, 
    eval_dataloader, 
    accelerator, 
    device, 
    logger, 
    n_gpu,  
    args, 
    metrics, 
    label_list, 
    label_to_sentinel_token=None,
    write_predictions=False,
    constrained_decoding=False, 
    mapping=None):

    if constrained_decoding and mapping is None:
        raise ValueError("For constrained decoding, set the mapping parameter")
    if constrained_decoding:
        raise ValueError("Not Implemented")

    # metric_rouge, metric_bleu = metrics
    model.eval()
    if args.val_max_target_length is None:
        args.val_max_target_length = args.max_target_length

    if 'mamba' not in args.model_name_or_path:
        gen_kwargs = {
            "max_length": args.val_max_target_length if args is not None else config.max_length,
            "num_beams": args.num_beams,
            "output_attentions": True
        }
    else:
        gen_kwargs = {
            "max_length": args.val_max_target_length if args is not None else config.max_length
        }

    all_inputs, all_preds, all_labels = [], [], []

    for _, batch in tqdm(enumerate(eval_dataloader),total=math.ceil(len(eval_dataloader)/args.gradient_accumulation_steps),
                            disable=not accelerator.is_local_main_process if args.use_accelerate else False):
        assert batch['input_ids_lens'].shape[0]==1
        input_ids_len = int(batch['input_ids_lens'][0])
        with torch.no_grad():
            if args.use_accelerate:
                generated_tokens = accelerator.unwrap_model(model).generate(
                    batch["input_ids"][0][:input_ids_len].view(1,-1),
                    # attention_mask=batch["attention_mask"][0],
                    **gen_kwargs,
                )
                generated_tokens = accelerator.pad_across_processes(
                    generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
                )

                labels = batch["labels"][0]
                inputs = batch["input_ids"][0]
                # if not args.pad_to_max_length: # If we did not pad to max length, we need to pad the labels too
                #     labels = accelerator.pad_across_processes(batch["labels"], dim=1, pad_index=tokenizer.pad_token_id)
                #     inputs = accelerator.pad_across_processes(batch["input_ids"], dim=1, pad_index=tokenizer.pad_token_id)
                inputs = accelerator.gather(inputs).cpu().numpy()[:input_ids_len]
                generated_tokens = accelerator.gather(generated_tokens[0]).cpu().numpy()[input_ids_len:]
                labels = accelerator.gather(labels).cpu().numpy()[input_ids_len:]

                # print(inputs)
                # print(generated_tokens)
                # print(labels)
                # print(tokenizer.decode(inputs, skip_special_tokens=True))
                # print(tokenizer.decode(labels, skip_special_tokens=True))
                # print(tokenizer.decode(generated_tokens, skip_special_tokens=True))
                # exit()

            else:
                if args.no_cuda or n_gpu==1:
                    generated_tokens = model.generate(
                        batch["input_ids"][0][:input_ids_len].view(1,-1).to(device),
                        # attention_mask=batch["attention_mask"][0].to(device),
                        **gen_kwargs,
                    )
                elif n_gpu>1:
                    generated_tokens = model.module.generate(
                        batch["input_ids"][0][:input_ids_len].view(1,-1).to(device),
                        # attention_mask=batch["attention_mask"].to(device),
                        **gen_kwargs,
                    )
                
                generated_tokens = generated_tokens.cpu().numpy()[input_ids_len:]   
                labels = batch["labels"][0][input_ids_len:]
                inputs = batch["input_ids"][0].numpy()[:input_ids_len]

            if isinstance(generated_tokens, tuple):
                generated_tokens = generated_tokens[0]
            if args.ignore_pad_token_for_loss:
                # Replace -100 in the labels as we can't decode them.
                labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

            decoded_inputs = tokenizer.decode(inputs, skip_special_tokens=True if label_to_sentinel_token is None else False)
            decoded_preds = tokenizer.decode(generated_tokens, skip_special_tokens=True if label_to_sentinel_token is None else False)
            decoded_labels = tokenizer.decode(labels, skip_special_tokens=True if label_to_sentinel_token is None else False)
            
            # decoded_preds, decoded_labels, decoded_tok_preds, decoded_tok_labels, decoded_r_preds, decoded_r_labels = postprocess_text([decoded_preds], [decoded_labels])

            all_inputs += [decoded_inputs]
            all_preds += [decoded_preds]
            all_labels += [decoded_labels]

    if write_predictions: # save predictions 
        if not args.use_accelerate:
            write_preds(inputs=all_inputs, predictions=all_preds, references=all_labels, filename=args.pred_file)
        elif accelerator.is_local_main_process: 
            write_preds(inputs=all_inputs, predictions=all_preds, references=all_labels, filename=args.pred_file)
    
    y_true, y_pred = [], []
    for i,g,p in zip(all_inputs,all_labels,all_preds):
        input_tokens = i.split('</s>')[0]
        input_tokens = re.split(" |'",input_tokens)
        if label_to_sentinel_token is None:
            y_true += [process_outputs(input_tokens, g, label_list, constrained_decoding=constrained_decoding)]
            y_pred += [process_outputs(input_tokens, p, label_list, constrained_decoding=constrained_decoding)]
        else:
            y_true += [process_outputs_text_infilling(input_tokens, g, label_to_sentinel_token, is_v8=True if args.prefix_scheme in ['v8','v9'] else False)]
            y_pred += [process_outputs_text_infilling(input_tokens, p, label_to_sentinel_token, is_v8=True if args.prefix_scheme==['v8','v9'] else False)]
            
            # print(input_tokens)
            # print(g)
            # print(p)
            # print(y_true[-1])
            # print(y_pred[-1])

    # for i in range(5):
    #     print(all_inputs[i])
    #     print(all_labels[i])
    #     print(all_preds[i])
    #     print(y_true[i])
    #     print(y_pred[i])
    # exit()
    
    ner_report = classification_report(y_true, y_pred, digits=4, zero_division=0)
    macro_f = f1_score(y_true, y_pred, average='macro')
    logger.info("\n%s", ner_report)

    # result_string = '\n'.join([str(result_rouge),'BLEU: {:.4f}'.format(result_bleu["bleu"]*100), ner_report])+'\n'
    result_string = ner_report+'\n'
    if not args.use_accelerate or accelerator.is_local_main_process:
        write_file(result_string,args.result_file)
    logger.info("***** Evaluation finished! *****")

    return macro_f, result_string
