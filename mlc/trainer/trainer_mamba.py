from transformers import Trainer
from transformers.trainer_utils import is_main_process
from transformers.utils import logging
import torch
import os

logger = logging.get_logger(__name__)

class MambaTrainer(Trainer):
    # def compute_loss(self, model, inputs, return_outputs=False):
    #     input_ids = inputs.pop("input_ids")
    #     lm_logits = model(input_ids).logits
    #     labels = input_ids.to(lm_logits.device)
    #     shift_logits = lm_logits[:, :-1, :].contiguous()
    #     labels = labels[:, 1:].contiguous()
    #     loss_fct = torch.nn.CrossEntropyLoss()
    #     lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), labels.view(-1))
    #     return lm_loss

    
    def save_model(self, output_dir=None, _internal_call=False):
        if output_dir is None:
            output_dir = self.args.output_dir 
        if is_main_process(self.args.local_rank):
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Saving model checkpoint to {output_dir}")
                
            torch.save(self.model.state_dict(), f"{output_dir}/pytorch_model.bin")
            self.tokenizer.save_pretrained(output_dir)
        