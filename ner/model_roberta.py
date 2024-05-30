import torch
from torch import nn
import torch.nn.functional as F
from dataclasses import dataclass
from transformers import RobertaForTokenClassification, RobertaConfig 
from transformers.file_utils import ModelOutput
from transformers.modeling_outputs import TokenClassifierOutput
from typing import Optional, Tuple, Union

class RobertaNer(RobertaForTokenClassification):

    def __init__(self, config: RobertaConfig, do_aux_task:bool=False):
        super(RobertaNer, self).__init__(config)
        if do_aux_task: 
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)
            self.aux_classifier = nn.Linear(config.hidden_size, 2)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        aux_labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        
        logits = self.classifier(sequence_output)

        if hasattr(self,'aux_classifier'):
            cls_token_output = sequence_output[:, 0, :]
            cls_token_output = self.dense(cls_token_output)
            cls_token_output = torch.tanh(cls_token_output)
            cls_token_output = self.dropout(cls_token_output)
            aux_logits = self.aux_classifier(cls_token_output)
            aux_logits = aux_logits.view(-1, 2)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                if hasattr(self,'aux_classifier'):
                    loss = 0.7*loss_fct(active_logits, active_labels)
                    loss += 0.3*loss_fct(aux_logits, aux_labels)
                else:
                    loss = loss_fct(active_logits, active_labels)
            else:
                if hasattr(self,'aux_classifier'):
                    loss = 0.7*loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                    loss += 0.3*loss_fct(aux_logits, aux_labels.view(-1))
                else:
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,aux_logits,) + outputs[2:] if hasattr(self,'aux_classifier') else (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            aux_logits=aux_logits if hasattr(self,'aux_classifier') else None,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        
class RobertaNerMtl(RobertaForTokenClassification):

    def __init__(self, config: RobertaConfig, num_presence_labels: int, num_period_labels: int):
        super(RobertaNerMtl, self).__init__(config)
        self.num_presence_labels = num_presence_labels
        self.num_period_labels = num_period_labels

        self.presence_classifier = nn.Linear(config.hidden_size, num_presence_labels)
        self.period_classifier = nn.Linear(config.hidden_size, num_period_labels)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        presence_labels = None, 
        period_labels = None, 
        label_weights = None, 
        presence_label_weights = None, 
        period_label_weights = None, 
        loss_weights = [1/3]*3,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)

        logits = self.classifier(sequence_output)
        presence_logits = self.presence_classifier(sequence_output)
        period_logits = self.period_classifier(sequence_output)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(weight = torch.FloatTensor(label_weights).cuda(), ignore_index=-100) 
            loss_fct_presence = nn.CrossEntropyLoss(weight = torch.FloatTensor(presence_label_weights).cuda() ,ignore_index=-100)
            loss_fct_period = nn.CrossEntropyLoss(weight = torch.FloatTensor(period_label_weights).cuda(),ignore_index=-100)
            loss = loss_weights[0]*loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            loss += loss_weights[1]*loss_fct_presence(presence_logits.view(-1, self.num_presence_labels), presence_labels.view(-1))
            loss += loss_weights[2]*loss_fct_period(period_logits.view(-1, self.num_period_labels), period_labels.view(-1))
            # return loss
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(weight = torch.FloatTensor(label_weights).cuda(), ignore_index=-100)
            loss_fct_presence = nn.CrossEntropyLoss(weight = torch.FloatTensor(presence_label_weights).cuda() ,ignore_index=-100)
            loss_fct_period = nn.CrossEntropyLoss(weight = torch.FloatTensor(period_label_weights).cuda(),ignore_index=-100)
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_labels = torch.where(active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels))
                active_presence_labels = torch.where(active_loss, presence_labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels))
                active_period_labels = torch.where(active_loss, period_labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels))
 
                loss = loss_weights[0]*loss_fct(logits.view(-1, self.num_labels), active_labels)
                loss += loss_weights[1]*loss_fct_presence(presence_logits.view(-1, self.num_presence_labels), active_presence_labels.view(-1))
                loss += loss_weights[2]*loss_fct_period(period_logits.view(-1, self.num_period_labels), active_period_labels.view(-1))
            else:
                loss = loss_weights[0]*loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                loss += loss_weights[1]*loss_fct_presence(presence_logits.view(-1, self.num_presence_labels), presence_labels.view(-1))
                loss += loss_weights[2]*loss_fct_period(period_logits.view(-1, self.num_period_labels), period_labels.view(-1))


        if not return_dict:
            output = (logits,presence_logits,period_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            presence_logits=presence_logits,
            period_logits=period_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

@dataclass
class TokenClassifierOutput(ModelOutput):
    """
    Base class for outputs of custom token classification models.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    aux_logits: torch.FloatTensor = None
    presence_logits: torch.FloatTensor = None
    period_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None