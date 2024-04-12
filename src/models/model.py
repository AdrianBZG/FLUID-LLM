import logging

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer, AutoModelForCausalLM
import transformers
from peft import LoraConfig, get_peft_model
from cprint import c_print
from collections import deque
from contextlib import nullcontext

from utils import freeze_model, unfreeze_model
from models.layers.input_embeddings import InputEmbeddings
from models.layers.patch_decoder import PatchDecoder
from models.layers.passthrough_embeddings import PassthroughEmbeddings

transformers.logging.set_verbosity_error()

logging.basicConfig(level=logging.INFO,
                    format=f'[{__name__}:%(levelname)s] %(message)s')


class MultivariateTimeLLM(nn.Module):
    def __init__(self, config, device_map='cpu'):
        super().__init__()

        self.config = config
        self.task_name = config['task_name']
        self.autocast_ctx = torch.cuda.amp.autocast(dtype=torch.bfloat16) if config['half_precision'] else nullcontext()

        # Get LLM backbone config and adapt appropriately
        # Ex.: huggyllama/llama-7b, openai-community/gpt2, google-bert/bert-base-uncased
        llm_config = AutoConfig.from_pretrained(config['llm_backbone'])
        if config['llm_layers'] > llm_config.num_hidden_layers:
            raise ValueError(f"Requested number of layers ({config['llm_layers']}) is greater than the model's ({llm_config.num_hidden_layers})!")

        llm_config.num_hidden_layers = config['llm_layers'] if config['llm_layers'] > 0 else llm_config.num_hidden_layers
        llm_config.output_attentions = True
        llm_config.output_hidden_states = True
        self.llm_config = llm_config

        self.backbone = AutoModel.from_pretrained(
            pretrained_model_name_or_path=config['llm_backbone'],
            trust_remote_code=True,
            local_files_only=False,
            config=self.llm_config,
            load_in_4bit=config['llm_4bit_loading'],
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            attn_implementation="flash_attention_2" if config['flash_attention'] else "eager",
        )

        c_print(f'LLM config: {llm_config}', color='green')

        # BOS token if needed
        if config['use_bos_token']:
            self.tokenizer = AutoTokenizer.from_pretrained(
                config['llm_backbone'],
                trust_remote_code=True,
                local_files_only=False
            )

            # Get the BOS token
            BOS_id = self.tokenizer.bos_token_id
            embeddings = self.backbone.get_input_embeddings()
            BOS_embed = embeddings(torch.tensor(BOS_id).to(device_map)).clone()
            self.BOS_embed = torch.nn.Parameter(BOS_embed)

        self.llm_in_dim = self.backbone.get_input_embeddings().weight.shape[1]

        self.N_x_patch, self.N_y_patch = config["patch_size"]
        self.patch_in_dim = self.N_x_patch * self.N_y_patch * 3
        self.max_seq_len = self.config['seq_len'] - 1

        # Input and output embeddings
        self.input_embeddings = InputEmbeddings(self.patch_in_dim,
                                                self.llm_in_dim,
                                                self.config['encoder_params'],
                                                config['input_emb_layer_dropout'],
                                                self.config['input_emb_layer_norm_eps'],  # self.llm_config.layer_norm_epsilon,
                                                self.config['max_num_embed'],
                                                pos_embedding_type=config['pos_embedding_type'],
                                                init_pos_embed=config['init_pos_embed'],
                                                use_self_attn=config['use_patches_self_attention'])

        self.output_layer = PatchDecoder(self.llm_in_dim, self.patch_in_dim, self.config['decoder_params'])

        # Adjust the backbone for time series task
        self._adjust_backbone()
        self.to(device_map)

        self.device_map = device_map

    def _adjust_backbone(self):
        # Nullify undesired layers
        self.backbone.embeddings = PassthroughEmbeddings()

        if not self.config['freeze_llm']:
            if self.config['use_lora']:
                logging.info(f"Using LoRA with config: {self.config['lora_config']}")
                config = LoraConfig(**self.config['lora_config'])
                self.backbone = get_peft_model(self.backbone, config)
                self.backbone.print_trainable_parameters()
            else:
                logging.info(f"Fine-tuning the entire LLM without LoRA.")
        else:
            # Freeze backbone parameters
            freeze_model(self.backbone)

    def forward(self, x, position_ids):
        batch_size = x.shape[0]

        # Encode with patch embedder
        x_enc = self.input_embeddings(x, position_ids)
        if self.config['use_bos_token']:
            x_enc = torch.cat([self.BOS_embed.unsqueeze(0).expand(batch_size, -1, -1), x_enc], dim=1)
            backbone_out = self.backbone(inputs_embeds=x_enc)
            backbone_preds = backbone_out.last_hidden_state[:, 1:]
        else:
            # Pass through frozen LLM
            backbone_out = self.backbone(inputs_embeds=x_enc)
            backbone_preds = backbone_out.last_hidden_state

        # Decode hidden state given by the LLM
        _, seq_len, _ = backbone_preds.shape
        decoder_out = self.output_layer(backbone_preds)
        decoder_out = decoder_out.view(batch_size, seq_len, 3, self.N_x_patch, self.N_y_patch)

        return backbone_out, decoder_out * self.config['diff_scale_factor']

    def _gen_step(self, states, position_ids, N_patch):
        """ Generate next timestep of the sequence given an input sequence.
            Use last given timestep as initialisation to generate diffs for next step
            Input.shape = (bs, seq_len*N_patch, 3, 16, 16)
            Return.shape = (bs, (seq_len+1)*N_patch, 3, 16, 16)"""

        _, pred_diff = self.forward(states, position_ids)
        diffs = pred_diff[:, -N_patch:]
        return diffs

    def _generate(self, init_states, bc_mask, position_ids, N_patch, N_steps):
        """ Given an input step(s), generate the next step(s) using the model.
        N_patch: Number of patches in each state
        N_steps: Number of steps to predict

        Keep 2 buffers, one for all states / diffs, and one for sliding model input.
        Ensure model input isn't too long and normalise timesteps to start at 0·

        init_states.shape = (bs, init_len*N_patch, 3, 16, 16)
        all_states.shape = (bs, (init_len+N_steps)*N_patch, 3, 16, 16)
        all_diffs.shape = (bs, N_steps*N_patch, 3, 16, 16)
        """

        # All states and diffs, including input and predictions for output.
        init_states = init_states.to(torch.float32)
        all_states = [init_states]
        all_diffs = []
        # Keep a buffer of the last N states as model input
        init_states_t = init_states.view(init_states.shape[0], -1, N_patch, 3, self.N_x_patch, self.N_y_patch)
        init_len = init_states_t.shape[1]
        input_buff = deque(maxlen=self.max_seq_len)
        for t in range(init_len):
            input_buff.append(init_states_t[:, t])

        for pred_step in range(init_len, init_len+N_steps):
            # print(f'{pred_step = }')
            seq_len = len(input_buff)
            # Get correct position ids
            end_pos = pred_step * N_patch
            start_pos = (pred_step - seq_len) * N_patch
            seq_pos_ids = position_ids[:, start_pos:end_pos].clone()       # shape = [bs, seq_len*N_patch, 3, ...]
            # Normalise timestep so first state is t=0
            min_t = seq_pos_ids[:, :, 2].min()
            seq_pos_ids[:, :, 2] = seq_pos_ids[:, :, 2] - min_t

            # Get masks for current state
            mask = bc_mask[:, end_pos - N_patch: end_pos]    # shape = [bs, N_patch, 3, ...]

            s = torch.cat(list(input_buff), dim=1)
            diffs = self._gen_step(s, seq_pos_ids, N_patch)
            diffs[mask] = 0.

            # Calculate diffs in fp32
            diffs = diffs.to(torch.float32)
            all_diffs.append(diffs)

            # Add on next state
            next_state = input_buff[-1] + diffs
            all_states.append(next_state)
            input_buff.append(next_state)

        all_states = torch.cat(all_states, dim=1)
        all_diffs = torch.cat(all_diffs, dim=1)
        return all_states, all_diffs

    def gen_seq(self, batch_data, N_patch, pred_steps, start_state=1):
        """ Evaluate the model by generating the next steps in the sequence."""
        states, _, bc_mask, position_ids = batch_data
        position_ids, bc_mask = position_ids.to(self.device_map), bc_mask.to(self.device_map)

        tot_seq_len = bc_mask.shape[1] // N_patch
        assert pred_steps + start_state - 1 <= tot_seq_len, \
            f'Prediction steps ({pred_steps}) must be less than total sequence length ({tot_seq_len}) + 1!'

        # Make sure the model can see everything before making the first prediction, duplicate the first state if start=1
        if start_state == 1:
            states = torch.cat([states[:, :N_patch], states], dim=1)
            init_state = states[:, :2 * N_patch].to(self.device_map)
            bc_mask = torch.cat([bc_mask[:, :N_patch], bc_mask], dim=1)
            position_ids = torch.cat([position_ids[:, :N_patch], position_ids], dim=1)
        else:
            init_state = states[:, :start_state * N_patch].to(self.device_map)

        all_states, all_diffs = self._generate(init_state, bc_mask, position_ids, N_patch, pred_steps)

        if start_state == 1:
            all_states = all_states[:, N_patch:]

        return all_states, all_diffs

    def forward_duplicate(self, states, position_ids, N_patch):
        """ Repeat the first state so the model can see the entire initial conditions before making any predictions"""
        states = torch.cat([states[:, :N_patch], states], dim=1)
        position_ids = torch.cat([position_ids[:, :N_patch], position_ids], dim=1)
        _, preds = self.forward(states, position_ids)

        preds = preds[:, N_patch:]

        return preds
        # print(states.shape)
        # print(position_ids[0, :, 2])
        # exit(9)
        #


