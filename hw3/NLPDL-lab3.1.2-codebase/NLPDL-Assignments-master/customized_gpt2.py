from turtle import forward
from typing import Optional, Tuple, Union, List
import torch
import torch.utils.checkpoint
from torch import nn
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention, GPT2Block, GPT2Model, GPT2LMHeadModel

class CustomizedGPT2Attention(GPT2Attention):
    """
    GPT2 flash attention module. This module inherits from `GPT2Attention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        attention_mask: Optional[torch.FloatTensor] = None,
        **kwargs,
    ):

        # Prepare query, key, value matrix
        query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2) # each of them has shape (batch_size, seq_len, dim)
        query = self._split_heads(query, self.num_heads, self.head_dim) # [batch_size, num_heads, seq_len, head_dim]
        key = self._split_heads(key, self.num_heads, self.head_dim) # [batch_size, num_heads, seq_len, head_dim]
        value = self._split_heads(value, self.num_heads, self.head_dim) # [batch_size, num_heads, seq_len, head_dim]

        # Self-attention mechanism
        attn_output, attn_weights = self._attn(query, key, value, attention_mask)

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim) # [batch_size, seq_len, dim]
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        return attn_output

class CustomizedGPT2AttentionWithFasterCache(GPT2Attention):
    """
    GPT2 flash attention module. This module inherits from `GPT2Attention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        cached_k: Optional[torch.FloatTensor] = None,
        cached_v: Optional[torch.FloatTensor] = None,
        **kwargs,
    ):
        if cached_k is not None and cached_v is not None:
            hidden_states = hidden_states[:, -1:, :]
            hidden_states = hidden_states.reshape(hidden_states.shape[0], 1, -1)
        
        # Prepare query, key, value matrix
        query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

        # Split heads
        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        if cached_k is not None and cached_v is not None:
            key = torch.cat((cached_k, key), dim=2)
            value = torch.cat((cached_v, value), dim=2)

        # Self-attention mechanism
        attn_output, attn_weights = self._attn(query, key, value, attention_mask)

        # Merge heads and apply final projection
        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        return attn_output, key, value

class CustomizedGPT2Block(GPT2Block):
    def __init__(self, config, layer_idx=None):
        super().__init__(config, layer_idx=layer_idx)
        # self.attn = CustomizedGPT2Attention(config=config, layer_idx=layer_idx)
        self.attn = CustomizedGPT2AttentionWithFasterCache(config=config, layer_idx=layer_idx)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        cached_k: Optional[torch.FloatTensor] = None,
        cached_v: Optional[torch.FloatTensor] = None,
        **kwargs,
    ):
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        # Include layer_past in the attention layer forward method
        attn_output, new_k, new_v = self.attn(
            hidden_states,
            attention_mask=attention_mask,
            cached_k=cached_k,
            cached_v=cached_v,
            **kwargs
        )

        hidden_states = attn_output + residual
        residual = hidden_states

        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        hidden_states = residual + feed_forward_hidden_states

        # Return both hidden_states and newly computed key and value tensors
        return hidden_states, new_k, new_v

class CustomizedGPT2Model(GPT2Model):
    def __init__(self, config):
        super().__init__(config)
        self.h = nn.ModuleList([CustomizedGPT2Block(config, layer_idx=i) for i in range(config.num_hidden_layers)])
        self._attn_implementation = config._attn_implementation
        assert self._attn_implementation == 'eager', "[NLPDL ERROR] set _attn_implementation to either 'eager' or 'faster_cache' in this version"

        self.k_cache = [None for _ in range(config.num_hidden_layers)]
        self.v_cache = [None for _ in range(config.num_hidden_layers)]

        # Initialize weights and apply final processing
        self.post_init()

    def reset(self):
        self.k_cache = [None for _ in range(self.config.num_hidden_layers)]
        self.v_cache = [None for _ in range(self.config.num_hidden_layers)]

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.FloatTensor = None,
        **kwargs
    ):
        input_shape = input_ids.size()
        batch_size = input_ids.shape[0]
        device = input_ids.device

        if self.k_cache[0] is not None:
            if input_ids.shape[1] <= self.k_cache[0].shape[2]:
                self.reset()

        inputs_embeds = self.wte(input_ids)
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        if attention_mask is not None:
            attention_mask = attention_mask.view(batch_size, -1)
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min

        hidden_states = self.drop(hidden_states)

        for i, block in enumerate(self.h):
            cached_k, cached_v = self.k_cache[i], self.v_cache[i]
            hidden_states, new_k, new_v = block(
                hidden_states,
                attention_mask=attention_mask,
                cached_k=cached_k,
                cached_v=cached_v,
                **kwargs
            )
            self.k_cache[i] = new_k
            self.v_cache[i] = new_v

        hidden_states = self.ln_f(hidden_states)
        hidden_states = hidden_states.view((-1,) + input_shape[1:] + (hidden_states.size(-1),))

        return hidden_states


class CustomizedGPT2LMHeadModel(GPT2LMHeadModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.transformer = CustomizedGPT2Model(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.FloatTensor = None,
    ):
        hidden_states = self.transformer(
            input_ids,
            attention_mask=attention_mask,
        )

        lm_logits = self.lm_head(hidden_states)
        return {"logits": lm_logits}