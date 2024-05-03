from typing import List, Optional, Tuple, Union
import torch
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.cache_utils import Cache, DynamicCache
from torch import nn
from transformers.utils import logging
from peft import PeftModel
from transformers.models.codegen.modeling_codegen import (
    CodeGenAttention,
    CodeGenBlock,
    CodeGenModel,
    CodeGenMLP,
    CodeGenForCausalLM,
    CodeGenPreTrainedModel,
    CodeGenConfig
)
logger = logging.get_logger(__name__)

# Copied from transformers.models.gptj.modeling_gptj.create_sinusoidal_positions
def create_sinusoidal_positions(num_pos: int, dim: int) -> torch.Tensor:
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, dtype=torch.int64) / dim))
    sinusoid_inp = torch.einsum("i , j -> i j", torch.arange(num_pos, dtype=torch.int64).float(), inv_freq).float()
    return torch.cat((torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)), dim=1)


# Copied from transformers.models.gptj.modeling_gptj.rotate_every_two
def rotate_every_two(x: torch.Tensor) -> torch.Tensor:
    x1 = x[:, :, :, ::2]
    x2 = x[:, :, :, 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')


# Copied from transformers.models.gptj.modeling_gptj.apply_rotary_pos_emb
def apply_rotary_pos_emb(tensor: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
    sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    return (tensor * cos) + (rotate_every_two(tensor) * sin)

class BidirectionalCodeGenAttention(CodeGenAttention):
    def __init__(self, config):
        super().__init__(config)
        max_positions = config.max_position_embeddings
        self.register_buffer(
            "causal_mask",
            torch.ones((max_positions, max_positions), dtype=torch.bool).view(
                1, 1, max_positions, max_positions
            ),
            persistent=False
        )
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        self.embed_dim = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_attention_heads
        if self.head_dim * self.num_attention_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_attention_heads (got `embed_dim`: {self.embed_dim} and"
                f" `num_attention_heads`: {self.num_attention_heads})."
            )
        self.scale_attn = torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32)).to(torch.get_default_dtype())
        self.qkv_proj = nn.Linear(self.embed_dim, self.embed_dim * 3, bias=False)

        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.rotary_dim = config.rotary_dim
        pos_embd_dim = self.rotary_dim or self.embed_dim
        self.embed_positions = create_sinusoidal_positions(max_positions, pos_embd_dim)

class BidirectionalCodeGenBlock(CodeGenBlock):
    def __init__(self, config:CodeGenConfig):
        nn.Module.__init__(self)
        inner_dim = config.n_inner if config.n_inner is not None else 4 * config.n_embd
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = BidirectionalCodeGenAttention(config)
        self.mlp = CodeGenMLP(inner_dim, config)


class CodeGenBiModel(CodeGenModel):
    def __init__(self, config):
        CodeGenPreTrainedModel.__init__(self, config)
        self.embed_dim = config.n_embd
        self.vocab_size = config.vocab_size

        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([BidirectionalCodeGenBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)
        self.rotary_dim = min(config.rotary_dim, config.n_ctx // config.num_attention_heads)

        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    # def forward(
    #     self,
    #     input_ids: Optional[torch.LongTensor] = None,
    #     attention_mask: Optional[torch.FloatTensor] = None,
    #     position_ids: Optional[torch.LongTensor] = None,
    #     head_mask: Optional[torch.FloatTensor] = None,
    #     token_type_ids: Optional[torch.LongTensor] = None,
    #     inputs_embeds: Optional[torch.FloatTensor] = None,
    #     past_key_values: Optional[List[torch.FloatTensor]] = None,
    #     use_cache: Optional[bool] = None,
    #     output_attentions: Optional[bool] = None,
    #     output_hidden_states: Optional[bool] = None,
    #     return_dict: Optional[bool] = None,
    # ) -> Union[Tuple, BaseModelOutputWithPast]:
    #     output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    #     output_hidden_states = (
    #         output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    #     )
    #     use_cache = use_cache if use_cache is not None else self.config.use_cache
    #     return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    #     if input_ids is not None and inputs_embeds is not None:
    #         raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
    #     elif input_ids is not None:
    #         self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
    #         input_shape = input_ids.size()
    #         input_ids = input_ids.view(-1, input_shape[-1])
    #         batch_size = input_ids.shape[0]
    #     elif inputs_embeds is not None:
    #         input_shape = inputs_embeds.size()[:-1]
    #         batch_size = inputs_embeds.shape[0]
    #     else:
    #         raise ValueError("You have to specify either input_ids or inputs_embeds")

    #     device = input_ids.device if input_ids is not None else inputs_embeds.device

    #     if position_ids is None:
    #         position_ids = torch.arange(input_shape[-1], dtype=torch.long, device=device)
    #         position_ids = position_ids.unsqueeze(0)

    #     # Attention mask.
    #     if attention_mask is not None:
    #         if batch_size <= 0:
    #             raise ValueError("batch_size has to be defined and > 0")
    #         attention_mask = attention_mask.view(batch_size, 1, 1, -1)  # Expand mask to 4D
    #         # Fill mask with large negative value where attention is not allowed
    #         attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
    #         attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min

    #     if inputs_embeds is None:
    #         inputs_embeds = self.wte(input_ids)

    #     hidden_states = self.drop(inputs_embeds)

    #     output_shape = input_shape + (hidden_states.size(-1),)

    #     if self.gradient_checkpointing and self.training:
    #         if use_cache:
    #             logger.warning_once(
    #                 "`use_cache=True` is incompatible with `config.gradient_checkpointing=True`. Setting "
    #                 "`use_cache=False`..."
    #             )
    #             use_cache = False

    #     presents = () if use_cache else None
    #     all_self_attentions = () if output_attentions else None
    #     all_hidden_states = () if output_hidden_states else None

    #     # Encoder blocks forward pass
    #     for block in self.h:
    #         if output_hidden_states:
    #             all_hidden_states = all_hidden_states + (hidden_states,)

    #         # Applying the encoder block
    #         hidden_states = block(
    #             hidden_states=hidden_states,
    #             attention_mask=attention_mask,
    #             position_ids=position_ids,
    #             head_mask=head_mask,
    #         )

    #     hidden_states = self.ln_f(hidden_states)

    #     if not return_dict:
    #         return hidden_states, presents, all_hidden_states, all_self_attentions
    #     return BaseModelOutputWithPast(
    #         last_hidden_state=hidden_states,
    #         past_key_values=presents,
    #         hidden_states=all_hidden_states,
    #         attentions=all_self_attentions,
    #     )

class BiCodeGenMNTP(CodeGenForCausalLM):
    def __init__(self, config, attention_dropout=0.0):
        CodeGenPreTrainedModel.__init__(self, config)
        self.transformer = CodeGenBiModel(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)
        # Initialize weights and apply final processing
        self.post_init()

    # getter for PEFT model
    def get_model_for_peft(self):
        return self.transformer

    # setter for PEFT model
    def set_model_for_peft(self, model: PeftModel):
        self.transformer = model

    # save the PEFT model
    def save_peft_model(self, path):
        self.transformer.save_pretrained(path)
