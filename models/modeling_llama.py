# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Callable, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn

from copy import copy

from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from transformers.processing_utils import Unpack

from transformers.utils import LossKwargs, auto_docstring, can_return_tuple, logging
from transformers.models.llama.modeling_llama import (
    LlamaPreTrainedModel,
    LlamaAttention,
    LlamaMLP,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    eager_attention_forward
)
from transformers.models.llama.modeling_llama import LlamaForCausalLM as LlamaForCausalLM_hf
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.masking_utils import create_causal_mask

# from ...integrations import use_kernel_forward_from_hub
# from ...modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
# from ...pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel

logger = logging.get_logger(__name__)
class KwargsForCausalLM(FlashAttentionKwargs, LossKwargs): ...


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len
    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
    inverted_mask = 1.0 - expanded_mask
    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


# [note] modify the kv hidden
class LlamaCrossAttention(LlamaAttention):
    """ The encoded rotary embeddings are done once in encoding. skip during the cross-attention """

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor, 
        attention_mask: Optional[torch.Tensor],
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        # reshape states 
        q_input_shape = hidden_states.shape[:-1]
        k_input_shape = encoder_hidden_states.shape[:-1]
        hidden_shape = (*q_input_shape, -1, self.head_dim)
        encoder_hidden_shape = (*k_input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(encoder_hidden_states).view(encoder_hidden_shape).transpose(1, 2)
        value_states = self.v_proj(encoder_hidden_states).view(encoder_hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings

        # reshape mask
        # attention_mask = attention_mask.view()

        # query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # [note] skip the key value caching (i) done when encoding (ii) document-level position
        # [note] CEPE: more memory consumption but slightly faster
        # if past_key_value is not None:
        #     # sin and cos are specific to RoPE models; cache_position needed for the static cache
        #     cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        #     key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        # [note] attention_mask might be incorrect
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*q_input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

# [code]
class LlamaDecoderLayer(GradientCheckpointingLayer):
    """ Add llama cross attention """
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = LlamaAttention(config=config, layer_idx=layer_idx)

        self.do_cross_attention = getattr(config, "do_cross_attention", True)
        if self.do_cross_attention:
            self.cross_attn = LlamaCrossAttention(config=config, layer_idx=layer_idx)

        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        do_cross_attention: bool = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Cross Attention
        enc_dec_attention_mask = encoder_attention_mask # this might be customized
        do_cross_attention = (do_cross_attention or self.do_cross_attention)
        if do_cross_attention and encoder_hidden_states is not None:
            residual = hidden_states
            bsz = hidden_states.size(0)
            _, _, tgt_len, src_len = enc_dec_attention_mask.shape # 6 1 2 45
            """
            hidden_states: (batch_size, (src)seq_length, hidden) 
            encoder_hidden_states: (batch_size, ctx_size * (tgt)seq_length, hidden) 
            cross attention mask: (N, 1, tgt_len, src_len) 
                - tgt: sequence length of causal decoded tokens 
                - src: sequence length of encoded embedding 

            # enc_dec_attention_mask = torch.randint(0, 2, (bsz, 1, 2, 78), dtype=torch.bool).to(hidden_states.device)# (N, *, tgt_len, src_len)
            """
            enc_dec_attention_mask = enc_dec_attention_mask.view(bsz, -1, tgt_len, src_len)
            hidden_states, cross_attn_weights = self.cross_attn(
                    hidden_states=hidden_states,
                    attention_mask=enc_dec_attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=False,
                    padding_mask=encoder_padding_mask,
            )
            hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs

# revised
@auto_docstring
class LlamaModel(LlamaPreTrainedModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)

        self.do_cross_attention = getattr(config, "do_cross_attention", False)
        self.num_cross_attn_layers = getattr(config, "num_cross_attn_layers", 8)
        # self.num_cross_attn_hidden_states = getattr(config, "num_cross_attn_hidden_states", 1)
        self.is_decoder = getattr(config, "is_decoder", True)

        layer_list = []
        for layer_idx in range(config.num_hidden_layers):
            config.do_cross_attention = (layer_idx >= config.num_hidden_layers - self.num_cross_attn_layers) and self.do_cross_attention
            layer_list.append(LlamaDecoderLayer(config, layer_idx))
        self.layers = nn.ModuleList(layer_list)

        config.do_cross_attention = self.do_cross_attention

        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        encoder_padding_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        do_cross_attention: bool = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> BaseModelOutputWithPast:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        B, L = input_ids.size()
        N = 1

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        # TODO (joao): remove this exception in v4.56 -- it exists for users that try to pass a legacy cache
        if not isinstance(past_key_values, (type(None), Cache)):
            raise ValueError("The `past_key_values` should be either a `Cache` object or `None`.")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # [NOTE] adjusst mask with encoder's hidden states 
        if encoder_hidden_states is not None:
            # if 0 in encoder_attention_mask:
            #     encoder_attention_mask_padding = encoder_attention_mask
            # else:
            #     encoder_attention_mask_padding = None
            encoder_attention_mask = encoder_attention_mask.view(B, -1)
            encoder_attention_mask = _expand_mask(encoder_attention_mask, inputs_embeds.dtype, tgt_len=L).to(inputs_embeds.device)

        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
        )
        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for idx, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):

            # if encoder_hidden_states is not None:
            #     print( (idx >= self.config.num_hidden_layers - self.num_cross_attn_layers) and self.do_cross_attention, encoder_hidden_states.shape)

            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # [note] keyerror happended if there is not idx's past-key-value
            try:
                past_key_value = past_key_values[idx]
            except:
                past_key_value = None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, past_key_value, output_attentions, padding_mask=padding_mask, is_causal=self.is_decoder)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer), 
                    hidden_states, attention_mask, 
                    encoder_hidden_state, encoder_attention_mask, 
                    encoder_padding_mask, position_ids,
                    do_cross_attention=(idx >= self.config.num_hidden_layers - self.num_cross_attn_layers) and self.do_cross_attention
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    encoder_padding_mask=encoder_padding_mask,
                    position_ids=position_ids,
                    do_cross_attention=(idx >= self.config.num_hidden_layers - self.num_cross_attn_layers) and self.do_cross_attention,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **flash_attn_kwargs,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


@auto_docstring
class LlamaForCausalLM(LlamaForCausalLM_hf):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(
        self, 
        config, 
        do_cross_attention=False,
        num_cross_attn_layers=8,
    ):
        super().__init__(config)
        # additional paramters
        config.do_cross_attention = do_cross_attention
        config.num_cross_attn_layers = num_cross_attn_layers

        self.model = LlamaModel(config)

        encoder_config = copy(config)
        encoder_config.do_cross_attention = False

        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.hidden_size = config.hidden_size

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_input_ids: Optional[torch.LongTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> CausalLMOutputWithPast:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        # [todo] add encoder
        if encoder_hidden_states is None:
            bsz, csz, seq_length = encoder_input_ids.shape
            encoder_input_ids = encoder_input_ids.view(bsz*csz, -1)
            encoder_hidden_states = self.model(
                input_ids=encoder_input_ids, 
                attention_mask=encoder_attention_mask, 
                do_cross_attention=False
            ).last_hidden_state

            encoder_hidden_states = encoder_hidden_states.view(bsz, -1, self.config.hidden_size) # (bsz, csz*seq_length, hidden)

        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            do_cross_attention=True,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

