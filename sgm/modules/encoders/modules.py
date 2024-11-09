import math
from contextlib import nullcontext
from functools import partial
from typing import Dict, List, Optional, Tuple, Union, Callable, Any

import kornia
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, repeat
from omegaconf import ListConfig
from diffusers.models.attention_processor import Attention
from diffusers.models.attention import FeedForward
from torch.utils.checkpoint import checkpoint
from transformers import (
    T5EncoderModel,
    T5Tokenizer,
)

from ...util import (
    append_dims,
    autocast,
    count_params,
    default,
    disabled_train,
    expand_dims_like,
    instantiate_from_config,
)


class AbstractEmbModel(nn.Module):
    def __init__(self):
        super().__init__()
        self._is_trainable = None
        self._ucg_rate = None
        self._input_key = None

    @property
    def is_trainable(self) -> bool:
        return self._is_trainable

    @property
    def ucg_rate(self) -> Union[float, torch.Tensor]:
        return self._ucg_rate

    @property
    def input_key(self) -> str:
        return self._input_key

    @is_trainable.setter
    def is_trainable(self, value: bool):
        self._is_trainable = value

    @ucg_rate.setter
    def ucg_rate(self, value: Union[float, torch.Tensor]):
        self._ucg_rate = value

    @input_key.setter
    def input_key(self, value: str):
        self._input_key = value

    @is_trainable.deleter
    def is_trainable(self):
        del self._is_trainable

    @ucg_rate.deleter
    def ucg_rate(self):
        del self._ucg_rate

    @input_key.deleter
    def input_key(self):
        del self._input_key


class GeneralConditioner(nn.Module):
    OUTPUT_DIM2KEYS = {2: "vector", 3: "crossattn", 4: "pl_emb", 5: "pl_emb"}
    KEY2CATDIM = {"vector": 1, "crossattn": 2, "concat": 1}

    def __init__(self, emb_models: Union[List, ListConfig], cor_embs=[], cor_p=[]):
        super().__init__()
        embedders = []
        for n, embconfig in enumerate(emb_models):
            embedder = instantiate_from_config(embconfig)
            assert isinstance(
                embedder, AbstractEmbModel
            ), f"embedder model {embedder.__class__.__name__} has to inherit from AbstractEmbModel"
            embedder.is_trainable = embconfig.get("is_trainable", False)
            embedder.ucg_rate = embconfig.get("ucg_rate", 0.0)
            if not embedder.is_trainable:
                embedder.train = disabled_train
                for param in embedder.parameters():
                    param.requires_grad = False
                embedder.eval()
            print(
                f"Initialized embedder #{n}: {embedder.__class__.__name__} "
                f"with {count_params(embedder, False)} params. Trainable: {embedder.is_trainable}"
            )

            if "input_key" in embconfig:
                embedder.input_key = embconfig["input_key"]
            elif "input_keys" in embconfig:
                embedder.input_keys = embconfig["input_keys"]
            else:
                raise KeyError(f"need either 'input_key' or 'input_keys' for embedder {embedder.__class__.__name__}")

            embedder.legacy_ucg_val = embconfig.get("legacy_ucg_value", None)
            if embedder.legacy_ucg_val is not None:
                embedder.ucg_prng = np.random.RandomState()

            embedders.append(embedder)
        self.embedders = nn.ModuleList(embedders)

        if len(cor_embs) > 0:
            assert len(cor_p) == 2 ** len(cor_embs)
        self.cor_embs = cor_embs
        self.cor_p = cor_p

    def possibly_get_ucg_val(self, embedder: AbstractEmbModel, batch: Dict) -> Dict:
        assert embedder.legacy_ucg_val is not None
        p = embedder.ucg_rate
        val = embedder.legacy_ucg_val
        for i in range(len(batch[embedder.input_key])):
            if embedder.ucg_prng.choice(2, p=[1 - p, p]):
                batch[embedder.input_key][i] = val
        return batch

    def surely_get_ucg_val(self, embedder: AbstractEmbModel, batch: Dict, cond_or_not) -> Dict:
        assert embedder.legacy_ucg_val is not None
        val = embedder.legacy_ucg_val
        for i in range(len(batch[embedder.input_key])):
            if cond_or_not[i]:
                batch[embedder.input_key][i] = val
        return batch

    def get_single_embedding(
        self,
        embedder,
        batch,
        output,
        cond_or_not: Optional[np.ndarray] = None,
        force_zero_embeddings: Optional[List] = None,
    ):
        embedding_context = nullcontext if embedder.is_trainable else torch.no_grad
        with embedding_context():
            if hasattr(embedder, "input_key") and (embedder.input_key is not None):
                if embedder.legacy_ucg_val is not None:
                    if cond_or_not is None:
                        batch = self.possibly_get_ucg_val(embedder, batch)
                    else:
                        batch = self.surely_get_ucg_val(embedder, batch, cond_or_not)
                emb_out = embedder(batch[embedder.input_key])
            elif hasattr(embedder, "input_keys"):
                emb_out = embedder(*[batch[k] for k in embedder.input_keys])
        assert isinstance(
            emb_out, (torch.Tensor, list, tuple)
        ), f"encoder outputs must be tensors or a sequence, but got {type(emb_out)}"
        if not isinstance(emb_out, (list, tuple)):
            emb_out = [emb_out]
        # print(f"?? {emb_out[0].shape} {emb_out[0].dtype}")
        for emb in emb_out:
            out_key = self.OUTPUT_DIM2KEYS[emb.dim()]
            if embedder.ucg_rate > 0.0 and embedder.legacy_ucg_val is None:
                assert cond_or_not == None
                if cond_or_not is None:
                    if "mask" in output.keys():
                        mask = output["mask"]
                    else:
                        mask = torch.bernoulli((1.0 - embedder.ucg_rate) * torch.ones(emb.shape[0], dtype=emb.dtype, device=emb.device))
                        output["mask"] = mask
                    
                    emb = (
                        expand_dims_like(
                            mask,
                            emb,
                        )
                        * emb
                    )
                    # emb = (
                    #     expand_dims_like(
                    #         torch.bernoulli((1.0 - embedder.ucg_rate) * torch.ones(emb.shape[0], dtype=emb.dtype, device=emb.device)),
                    #         emb,
                    #     )
                    #     * emb
                    # )
                    print(f"? ucg with embedder: {embedder.__class__.__name__}")
                    # print(f"? emb: {emb}")
                    # print(f"? mask: {expand_dims_like(torch.bernoulli((1.0 - embedder.ucg_rate) * torch.ones(emb.shape[0], dtype=emb.dtype, device=emb.device)), emb,)}")
                    print(f"mask: {mask}")
                else:
                    assert False
                    emb = (
                        expand_dims_like(
                            torch.tensor(1 - cond_or_not, dtype=emb.dtype, device=emb.device),
                            emb,
                        )
                        * emb
                    )

            # print(f"dtype in conditioner: {emb.dtype}")
            # print(f"shape in conditioner: {emb.shape}")
            emb = emb.to(torch.bfloat16)
            # print(f"dtype in conditioner: {emb.dtype}")
            # print(f"shape in conditioner: {emb.shape}")
            if hasattr(embedder, "input_key") and embedder.input_key in force_zero_embeddings:
                emb = torch.zeros_like(emb)
            if out_key in output:
                output[out_key] = torch.cat((output[out_key], emb), self.KEY2CATDIM[out_key])
            else:
                output[out_key] = emb
        return output

    def forward(self, batch: Dict, force_zero_embeddings: Optional[List] = None) -> Dict:
        output = dict()
        if force_zero_embeddings is None:
            force_zero_embeddings = []

        if len(self.cor_embs) > 0:
            assert False
            batch_size = len(batch[list(batch.keys())[0]])
            rand_idx = np.random.choice(len(self.cor_p), size=(batch_size,), p=self.cor_p)
            for emb_idx in self.cor_embs:
                cond_or_not = rand_idx % 2
                rand_idx //= 2
                output = self.get_single_embedding(
                    self.embedders[emb_idx],
                    batch,
                    output=output,
                    cond_or_not=cond_or_not,
                    force_zero_embeddings=force_zero_embeddings,
                )

        for i, embedder in enumerate(self.embedders):
            if i in self.cor_embs:
                continue
            output = self.get_single_embedding(
                embedder, batch, output=output, force_zero_embeddings=force_zero_embeddings
            )
        return output

    def get_unconditional_conditioning(self, batch_c, batch_uc=None, force_uc_zero_embeddings=None):
        if force_uc_zero_embeddings is None:
            force_uc_zero_embeddings = []
        ucg_rates = list()
        for embedder in self.embedders:
            ucg_rates.append(embedder.ucg_rate)
            embedder.ucg_rate = 0.0
        cor_embs = self.cor_embs
        cor_p = self.cor_p
        self.cor_embs = []
        self.cor_p = []

        # print(f"?batch_c: {batch_c.keys()} {batch_uc.keys()}")
        c = self(batch_c)
        uc = self(batch_c if batch_uc is None else batch_uc, force_uc_zero_embeddings)
        # print(f"? condition and unconditional: {c.keys()} {uc.keys()}")
        # print(f"shape of c['cross_attn']: {c['crossattn'].shape}")
        # print(f"shape of uc['cross_attn']: {uc['crossattn'].shape}")
        # print(f"shape of c['pl_emb']: {c['pl_emb'].shape}")
        # print(f"shape of uc['pl_emb']: {uc['pl_emb'].shape}")

        for embedder, rate in zip(self.embedders, ucg_rates):
            embedder.ucg_rate = rate
        self.cor_embs = cor_embs
        self.cor_p = cor_p

        return c, uc


class FrozenT5Embedder(AbstractEmbModel):
    """Uses the T5 transformer encoder for text"""

    def __init__(
        self,
        model_dir="google/t5-v1_1-xxl",
        device="cuda",
        max_length=77,
        freeze=True,
        cache_dir=None,
    ):
        super().__init__()
        if model_dir is not "google/t5-v1_1-xxl":
            self.tokenizer = T5Tokenizer.from_pretrained(model_dir)
            self.transformer = T5EncoderModel.from_pretrained(model_dir)
        else:
            self.tokenizer = T5Tokenizer.from_pretrained(model_dir, cache_dir=cache_dir)
            self.transformer = T5EncoderModel.from_pretrained(model_dir, cache_dir=cache_dir)
        self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()

        for param in self.parameters():
            param.requires_grad = False

    # @autocast
    def forward(self, text):
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=True,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )
        tokens = batch_encoding["input_ids"].to(self.device)
        with torch.autocast("cuda", enabled=False):
            outputs = self.transformer(input_ids=tokens)
        z = outputs.last_hidden_state
        return z

    def encode(self, text):
        return self(text)

# copy from CameraCtrl https://github.com/hehao13/CameraCtrl
def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")

class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(dims, self.channels, self.out_channels, 3, stride=stride, padding=padding)
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)

class ResnetBlock(nn.Module):

    def __init__(self, in_c, out_c, down, ksize=3, sk=False, use_conv=True):
        super().__init__()
        ps = ksize // 2
        if in_c != out_c or sk == False:
            self.in_conv = nn.Conv2d(in_c, out_c, ksize, 1, ps)
        else:
            self.in_conv = None
        self.block1 = nn.Conv2d(out_c, out_c, 3, 1, 1)
        self.act = nn.ReLU()
        self.block2 = nn.Conv2d(out_c, out_c, ksize, 1, ps)
        if sk == False:
            self.skep = nn.Conv2d(in_c, out_c, ksize, 1, ps)
        else:
            self.skep = None

        self.down = down
        if self.down == True:
            self.down_opt = Downsample(in_c, use_conv=use_conv)

    def forward(self, x):
        if self.down == True:
            x = self.down_opt(x)
        if self.in_conv is not None:  # edit
            x = self.in_conv(x)

        h = self.block1(x)
        h = self.act(h)
        h = self.block2(h)
        if self.skep is not None:
            return h + self.skep(x)
        else:
            return h + x

class PositionalEncoding(nn.Module):
    def __init__(
            self,
            d_model,
            dropout=0.,
            max_len=32,
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

# from CameraCtrl https://github.com/hehao13/CameraCtrl
import torch.nn.init as init
class PoseAdaptorAttnProcessor(nn.Module):
    def __init__(self,
                 hidden_size,  # dimension of hidden state
                 pose_feature_dim=None,  # dimension of the pose feature
                 cross_attention_dim=None,  # dimension of the text embedding
                 query_condition=False,
                 key_value_condition=False,
                 scale=1.0):
        super().__init__()

        self.hidden_size = hidden_size
        self.pose_feature_dim = pose_feature_dim
        self.cross_attention_dim = cross_attention_dim
        self.scale = scale
        self.query_condition = query_condition
        self.key_value_condition = key_value_condition
        assert hidden_size == pose_feature_dim
        if self.query_condition and self.key_value_condition:
            self.qkv_merge = nn.Linear(hidden_size, hidden_size)
            init.zeros_(self.qkv_merge.weight)
            init.zeros_(self.qkv_merge.bias)
        elif self.query_condition:
            self.q_merge = nn.Linear(hidden_size, hidden_size)
            init.zeros_(self.q_merge.weight)
            init.zeros_(self.q_merge.bias)
        else:
            self.kv_merge = nn.Linear(hidden_size, hidden_size)
            init.zeros_(self.kv_merge.weight)
            init.zeros_(self.kv_merge.bias)

    def forward(self,
                attn,
                hidden_states,
                pose_feature,
                encoder_hidden_states=None,
                attention_mask=None,
                temb=None,
                scale=None,):
        assert pose_feature is not None
        pose_embedding_scale = (scale or self.scale)

        residual = hidden_states
        # if attn.spatial_norm is not None:
        #     hidden_states = attn.spatial_norm(hidden_states, temb)

        if hidden_states.dim == 5:
            hidden_states = rearrange(hidden_states, 'b c f h w -> (b f) (h w) c')
        elif hidden_states.ndim == 4:
            hidden_states = rearrange(hidden_states, 'b c h w -> b (h w) c')
        else:
            assert hidden_states.ndim == 3

        if self.query_condition and self.key_value_condition:
            assert encoder_hidden_states is None

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        if encoder_hidden_states.ndim == 5:
            encoder_hidden_states = rearrange(encoder_hidden_states, 'b c f h w -> (b f) (h w) c')
        elif encoder_hidden_states.ndim == 4:
            encoder_hidden_states = rearrange(encoder_hidden_states, 'b c h w -> b (h w) c')
        else:
            assert encoder_hidden_states.ndim == 3
        if pose_feature.ndim == 5:
            pose_feature = rearrange(pose_feature, "b c f h w -> (b f) (h w) c")
        elif pose_feature.ndim == 4:
            pose_feature = rearrange(pose_feature, "b c h w -> b (h w) c")
        else:
            assert pose_feature.ndim == 3

        batch_size, ehs_sequence_length, _ = encoder_hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, ehs_sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        if attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        if self.query_condition and self.key_value_condition:  # only self attention
            query_hidden_state = self.qkv_merge(hidden_states + pose_feature) * pose_embedding_scale + hidden_states
            key_value_hidden_state = query_hidden_state
        elif self.query_condition:
            query_hidden_state = self.q_merge(hidden_states + pose_feature) * pose_embedding_scale + hidden_states
            key_value_hidden_state = encoder_hidden_states
        else:
            key_value_hidden_state = self.kv_merge(encoder_hidden_states + pose_feature) * pose_embedding_scale + encoder_hidden_states
            query_hidden_state = hidden_states

        # original attention
        query = attn.to_q(query_hidden_state)
        key = attn.to_k(key_value_hidden_state)
        value = attn.to_v(key_value_hidden_state)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

class TemporalSelfAttention(Attention):
    def __init__(
            self,
            attention_mode=None,
            temporal_position_encoding=False,
            temporal_position_encoding_max_len=32,
            rescale_output_factor=1.0,
            *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        assert attention_mode == "Temporal_Self"

        self.pos_encoder = PositionalEncoding(
            kwargs["query_dim"],
            max_len=temporal_position_encoding_max_len
        ) if temporal_position_encoding else None
        self.rescale_output_factor = rescale_output_factor

    def set_use_memory_efficient_attention_xformers(
            self, use_memory_efficient_attention_xformers: bool, attention_op: Optional[Callable] = None
    ):
        # disable motion module efficient xformers to avoid bad results, don't know why
        # TODO: fix this bug
        pass

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, **cross_attention_kwargs):
        # The `Attention` class can call different attention processors / attention functions
        # here we simply pass along all tensors to the selected processor class
        # For standard processors that are defined here, `**cross_attention_kwargs` is empty

        # add position encoding
        if self.pos_encoder is not None:
            hidden_states = self.pos_encoder(hidden_states)
        if "pose_feature" in cross_attention_kwargs:
            pose_feature = cross_attention_kwargs["pose_feature"]
            if pose_feature.ndim == 5:
                pose_feature = rearrange(pose_feature, "b c f h w -> (b h w) f c")
            else:
                assert pose_feature.ndim == 3
            cross_attention_kwargs["pose_feature"] = pose_feature

        if isinstance(self.processor,  PoseAdaptorAttnProcessor):
            return self.processor(
                self,
                hidden_states,
                cross_attention_kwargs.pop('pose_feature'),
                encoder_hidden_states=None,
                attention_mask=attention_mask,
                **cross_attention_kwargs,
            )
        elif hasattr(self.processor, "__call__"):
            return self.processor.__call__(
                    self,
                    hidden_states,
                    encoder_hidden_states=None,
                    attention_mask=attention_mask,
                    **cross_attention_kwargs,
                )
        else:
            return self.processor(
                self,
                hidden_states,
                encoder_hidden_states=None,
                attention_mask=attention_mask,
                **cross_attention_kwargs,
            )

class TemporalTransformerBlock(nn.Module):
    def __init__(
            self,
            dim,
            num_attention_heads,
            attention_head_dim,
            attention_block_types=("Temporal_Self", "Temporal_Self",),
            dropout=0.0,
            norm_num_groups=32,
            cross_attention_dim=768,
            activation_fn="geglu",
            attention_bias=False,
            upcast_attention=False,
            temporal_position_encoding=False,
            temporal_position_encoding_max_len=32,
            encoder_hidden_states_query=(False, False),
            attention_activation_scale=1.0,
            attention_processor_kwargs: Dict = {},
            rescale_output_factor=1.0
    ):
        super().__init__()

        attention_blocks = []
        norms = []
        self.attention_block_types = attention_block_types

        for block_idx, block_name in enumerate(attention_block_types):
            attention_blocks.append(
                TemporalSelfAttention(
                    attention_mode=block_name,
                    cross_attention_dim=cross_attention_dim if block_name in ['Temporal_Cross', 'Temporal_Pose_Adaptor'] else None,
                    query_dim=dim,
                    heads=num_attention_heads,
                    dim_head=attention_head_dim,
                    dropout=dropout,
                    bias=attention_bias,
                    upcast_attention=upcast_attention,
                    temporal_position_encoding=temporal_position_encoding,
                    temporal_position_encoding_max_len=temporal_position_encoding_max_len,
                    rescale_output_factor=rescale_output_factor,
                )
            )
            norms.append(nn.LayerNorm(dim))

        self.attention_blocks = nn.ModuleList(attention_blocks)
        self.norms = nn.ModuleList(norms)

        self.ff = FeedForward(dim, dropout=dropout, activation_fn=activation_fn)
        self.ff_norm = nn.LayerNorm(dim)

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, cross_attention_kwargs: Dict[str, Any] = {}):
        for attention_block, norm, attention_block_type in zip(self.attention_blocks, self.norms, self.attention_block_types):
            norm_hidden_states = norm(hidden_states)
            hidden_states = attention_block(
                norm_hidden_states,
                encoder_hidden_states=norm_hidden_states if attention_block_type == 'Temporal_Self' else encoder_hidden_states,
                attention_mask=attention_mask,
                **cross_attention_kwargs
            ) + hidden_states

        hidden_states = self.ff(self.ff_norm(hidden_states)) + hidden_states

        output = hidden_states
        return output

class PoseEmbedder(AbstractEmbModel):
    """Pose Encoder"""

    def __init__(
        self,
        downscale_factor: int,
        channels: List[int] =[320, 640, 1280, 1280],
        nums_rb: int = 3,
        cin=49,
        compression_factor=1,
        temporal_attention_nhead=8,
        attention_block_types=("Temporal_Self", ),
        use_conv=True,
        sk=False,
        ksize=3,
        temporal_position_encoding=False,
        temporal_position_encoding_max_len=16,
        rescale_output_factor=1.0,
        # freeze=False,
        device="cuda"
    ):
        super().__init__()
        
        self.unshuffle = nn.PixelUnshuffle(downscale_factor)
        self.channels = channels
        self.nums_rb = nums_rb
        self.encoder_down_conv_blocks = nn.ModuleList()
        self.encoder_down_attention_blocks = nn.ModuleList()

        self.downsample = Downsample(cin, use_conv=use_conv, dims=3)

        for i in range(len(channels)):
            conv_layers = nn.ModuleList()
            temporal_attention_layers = nn.ModuleList()
            for j in range(nums_rb):
                if j == 0 and i != 0:
                    in_dim = channels[i - 1]
                    out_dim = int(channels[i] / compression_factor)
                    conv_layer = ResnetBlock(in_dim, out_dim, down=True, ksize=ksize, sk=sk, use_conv=use_conv)
                elif j == 0:
                    in_dim = channels[0]
                    out_dim = int(channels[i] / compression_factor)
                    conv_layer = ResnetBlock(in_dim, out_dim, down=False, ksize=ksize, sk=sk, use_conv=use_conv)
                elif j == nums_rb - 1:
                    in_dim = int(channels[i] / compression_factor)
                    out_dim = channels[i]
                    conv_layer = ResnetBlock(in_dim, out_dim, down=False, ksize=ksize, sk=sk, use_conv=use_conv)
                else:
                    in_dim = int(channels[i] / compression_factor)
                    out_dim = int(channels[i] / compression_factor)
                    conv_layer = ResnetBlock(in_dim, out_dim, down=False, ksize=ksize, sk=sk, use_conv=use_conv)
                temporal_attention_layer = TemporalTransformerBlock(dim=out_dim,
                                                                    num_attention_heads=temporal_attention_nhead,
                                                                    attention_head_dim=int(out_dim / temporal_attention_nhead),
                                                                    attention_block_types=attention_block_types,
                                                                    dropout=0.0,
                                                                    cross_attention_dim=None,
                                                                    temporal_position_encoding=temporal_position_encoding,
                                                                    temporal_position_encoding_max_len=temporal_position_encoding_max_len,
                                                                    rescale_output_factor=rescale_output_factor)
                conv_layers.append(conv_layer)
                temporal_attention_layers.append(temporal_attention_layer)
            self.encoder_down_conv_blocks.append(conv_layers)
            self.encoder_down_attention_blocks.append(temporal_attention_layers)

        self.encoder_conv_in = nn.Conv2d(cin, channels[0], 3, 1, 1)
    
    def encode(self, x):
        # unshuffle
        bs = x.shape[0]
        x = rearrange(x, "b c f h w -> (b f) c h w")

        # downsample
        x = self.downsample(x)

        # shape of x: batch, c, frames, h, w
        x = self.unshuffle(x)
        # extract features
        features = []
        x = self.encoder_conv_in(x)
        for res_block, attention_block in zip(self.encoder_down_conv_blocks, self.encoder_down_attention_blocks):
            for res_layer, attention_layer in zip(res_block, attention_block):
                x = res_layer(x)
                h, w = x.shape[-2:]
                x = rearrange(x, '(b f) c h w -> (b h w) f c', b=bs)
                x = attention_layer(x)
                x = rearrange(x, '(b h w) f c -> (b f) c h w', h=h, w=w)
            features.append(x)
        return features

    def forward(self, pose_embedding):
        assert pose_embedding.ndim == 5
        bs = pose_embedding.shape[0]
        pose_embedding_features = self.encode(pose_embedding)      # bf c h w
        pose_embedding_features = [rearrange(x, '(b f) c h w -> b c f h w', b=bs)
                                   for x in pose_embedding_features]
        return pose_embedding_features[-1]
    
class PoseMLPEmbedder(AbstractEmbModel):
    """
    Pose Encoder using MLP and Conv
    [b, in_dim, f, h, w] -> [b, out_dim, f, h / 2, w / 2]
    f = frames
    h = video_size[0]
    w = video_size[1]
    downsample: (in_dim, h, w) -> (hidden_dim, h / 2, w / 2)
    # conv1: (in_dim, f, h / 2, w / 2) -> (hidden_dim, f, h / 2, w / 2)
    mlp1: (hidden_dim) -> (hidden_dim)
    conv2: (hidden_dim, f, h / 2, w / 2) -> (out_dim, f, h / 2, w / 2)
    """

    def __init__(
        self,
        in_channel: int = 6,
        out_channel: int = 3072,
        hidden_dim: int = 768,
        num_layers: int = 3,
        frames: int = 49,
        video_size: Tuple[int, int] = [480, 720],
        dropout: float = 0.5,
        device: str = "cuda",
    ):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.frames = frames
        self.video_size = video_size
        self.dropout = dropout
        self.device = device

        self.downsample = Downsample(in_channel, out_channels=hidden_dim, use_conv=True, dims=3)
        # self.conv1 = nn.Conv2d(in_channel, hidden_dim, 3, 1, 1)

        # self.mlp1 = nn.ModuleList()
        # for _ in range(num_layers):
        #     self.mlp1.append(nn.Sequential(
        #         nn.Linear(video_size[0] * video_size[1] // 4, video_size[0] * video_size[1] // 4),
        #         nn.ReLU(),
        #         nn.Dropout(dropout),
        #     ))

        self.mlp1 = nn.ModuleList()
        for _ in range(num_layers):
            self.mlp1.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ))

        self.conv2 = nn.Conv2d(hidden_dim, out_channel, 3, 1, 1)

    # def from_pretrained(self, model_dir):
    #     self.downsample = Downsample(self.in_channel, out_channels=self.hidden_dim, use_conv=True, dims=3)
    #     self.conv1 = nn.Conv2d(self.in_channel, self.hidden_dim, 3, 1, 1)
    #     self.mlp1 = nn.Sequential(
    #         nn.Linear(self.video_size[0] * self.video_size[1] // 4, self.video_size[0] * self.video_size[1] // 4),
    #         nn.ReLU(),
    #         nn.Linear(self.video_size[0] * self.video_size[1] // 4, self.video_size[0] * self.video_size[1] // 4),
    #         nn.ReLU(),
    #     )
    #     self.conv2 = nn.Conv2d(self.hidden_dim, self.out_channel, 3, 1, 1)
    #     self.load_state_dict(torch.load(model_dir))

    def forward(self, x):
        bs = x.shape[0]
        x = rearrange(x, "b c f h w -> (b f) c h w")
        x = self.downsample(x)
        # x = self.conv1(x)
        # x = rearrange(x, "(b f) c h w -> b c f (h w)", b=bs)
        x = rearrange(x, "(b f) c h w -> b f h w c", b=bs)
        x = self.mlp1(x)
        # x = rearrange(x, "b c f (h w) -> (b f) c h w", h=self.video_size[0] // 2)
        x = rearrange(x, "b f h w c -> (b f) c h w")
        x = self.conv2(x)
        x = rearrange(x, "(b f) c h w -> b c f h w", b=bs)
        return x
    
class PoseIdentityEmbedder(AbstractEmbModel):
    """Identity Pose Encoder"""

    def __init__(
        self,
        device: str = "cuda"
    ):
        super().__init__()
        self.device = device

    def forward(self, x):
        print(f"dtype in PoseIdentity Embedder {x.dtype}")
        print(f"shape in PoseIdentity Embedder {x.shape}")
        # (b f c h w)
        return x.to(self.device)