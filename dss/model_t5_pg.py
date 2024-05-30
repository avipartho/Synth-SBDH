import torch
from torch import nn
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration, T5Config 
from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutput
from typing import List
from gen_utils import GenerationMixin

class PromptEncoder(torch.nn.Module):
    def __init__(self, soft_prompt_len, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.seq_indices = torch.LongTensor(list(range(soft_prompt_len)))
        self.embedding = nn.Embedding(soft_prompt_len, self.hidden_size) # embedding layer

    def forward(self,device):
        input_embeds = self.embedding(self.seq_indices.to(device))
        return input_embeds

class t5_pg(GenerationMixin, T5ForConditionalGeneration):

    def __init__(self, config: T5Config, ner_tag_input_ids: List, use_pgen = True, soft_prompt_len=None, pseudo_token_id=None, tokenizer=None):
        super(t5_pg, self).__init__(config)
        self.ner_tag_input_ids = ner_tag_input_ids
        # self.alpha = 0.3
        if use_pgen:
            self.p_gen_linear = nn.Linear(config.d_model*2, 1)
        if soft_prompt_len is not None and soft_prompt_len> 0: 
            self.soft_prompt_len=soft_prompt_len
            self.prompt_encoder = PromptEncoder(soft_prompt_len, config.d_model)
            self.pseudo_token_id = pseudo_token_id
            self.tokenizer = tokenizer

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None
    ):

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                decoder_head_mask = head_mask

        # Soft prompt encoder
        if hasattr(self,'prompt_encoder'):
            inputs_embeds = input_ids.clone()
            inputs_embeds[(input_ids == self.pseudo_token_id)] = self.tokenizer.unk_token_id
            inputs_embeds = self.shared(inputs_embeds)  # (bsz,enc_seq_len,feat_dim)

            blocked_indices = torch.nonzero(input_ids == self.pseudo_token_id, as_tuple=False).reshape((inputs_embeds.shape[0], self.soft_prompt_len, 2))[:, :, 1]  # getting all the indices for soft prompts
            replace_embeds = self.prompt_encoder(device=input_ids.device)

            for b_idx in range(inputs_embeds.shape[0]):
                for i in range(self.soft_prompt_len):
                    inputs_embeds[b_idx, blocked_indices[b_idx, i], :] = replace_embeds[i, :]

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=None if hasattr(self,'prompt_encoder') else input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            sequence_output = sequence_output * (self.model_dim ** -0.5)

        lm_logits = self.lm_head(sequence_output) # (bsz,dec_seq_len,vocab_size)

        # attention distribution over source tokens
        # encoder_in_and_out = self.alpha*self.shared(input_ids)+(1-self.alpha)*encoder_outputs[0] # (bsz,enc_seq_len,feat_dim)
        source_token_attn_dist = torch.bmm(sequence_output,encoder_outputs[0].transpose(1,2)) # (bsz,dec_seq_len,feat_dim)X(bsz,feat_dim,enc_seq_len) >> (bsz,dec_seq_len,enc_seq_len)
        # source_token_attn_dist = decoder_outputs.cross_attentions[-1]  # (bsz,n_heads,dec_seq_len,enc_seq_len)
        # source_token_attn_dist = source_token_attn_dist.mean(dim=1)  # (bsz,dec_seq_len,enc_seq_len)

        # attention distribution over ner tags
        ner_tag_input_ids = torch.tensor(self.ner_tag_input_ids,device=source_token_attn_dist.device)
        ner_tag_input_embeds = self.shared(ner_tag_input_ids) # (num_ner_tags,feat_dim)
        ner_tag_attn_dist = torch.matmul(sequence_output,ner_tag_input_embeds.transpose(0,1)) # (bsz,dec_seq_len,feat_dim)X(feat_dim,num_ner_tags) >> (bsz,dec_seq_len,num_ner_tags)
        num_ner_tags = len(ner_tag_input_ids)

        # calculate p_gen
        if hasattr(self,'p_gen_linear'):
            source_context = torch.bmm(F.softmax(source_token_attn_dist,dim=2),encoder_outputs[0]) # (bsz,dec_seq_len,enc_seq_len)X(bsz,enc_seq_len,feat_dim) >> (bsz,dec_seq_len,feat_dim)
            p_gen_input = torch.cat((source_context,sequence_output),dim=2) # (bsz,dec_seq_len,2*feat_dim)
            p_gen = self.p_gen_linear(p_gen_input) # (bsz,dec_seq_len,1)
            p_gen = torch.sigmoid(p_gen)
            lm_logits *= p_gen # mulltiply vocab distribution by p_gen

        # mulltiply attention distributions by (1-p_gen) if necessary and add them to the vocab distribution
        source_token_indices = input_ids.unsqueeze(1).repeat(1,lm_logits.shape[1],1) # (bsz,enc_seq_len) >> (bsz,dec_seq_len,enc_seq_len)
        if hasattr(self,'p_gen_linear'):
            lm_logits[:,:,-num_ner_tags:] += (1-p_gen)*ner_tag_attn_dist # non-deterministic
            lm_logits.scatter_add_(2, source_token_indices, (1-p_gen)*source_token_attn_dist) # non-deterministic
        else:
            lm_logits[:,:,-num_ner_tags:] += ner_tag_attn_dist
            lm_logits.scatter_add_(2, source_token_indices, source_token_attn_dist) # non-deterministic
        
        # Does the same as line 138/141, but non-deterministic
        # ner_tag_indices = ner_tag_input_ids.unsqueeze(0).repeat(lm_logits.shape[1],1) # num_ner_tags >> (dec_seq_len,num_ner_tags)
        # ner_tag_indices = ner_tag_indices.unsqueeze(0).repeat(lm_logits.shape[0],1,1) # (dec_seq_len,num_ner_tags) >> (bsz,dec_seq_len,num_ner_tags)
        # lm_logits.scatter_add_(2, ner_tag_indices, (1-p_gen)*ner_tag_attn_dist)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            
        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past=None, attention_mask=None, use_cache=None, encoder_outputs=None, encoder_input_ids=None, **kwargs
    ):

        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]

        return {
            "input_ids": encoder_input_ids,
            "decoder_input_ids": input_ids,
            "past_key_values": past,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "use_cache": use_cache,
        }