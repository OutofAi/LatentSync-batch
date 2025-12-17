# Adapted from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention.py

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import BaseOutput
from diffusers.utils.import_utils import is_xformers_available
from diffusers.models.attention import Attention as CrossAttention, FeedForward, AdaLayerNorm

from einops import rearrange
from .utils import zero_module
import flash_attn_interface

print(f'torch:{torch.__version__}')

class FlashAttnProcessor2_0:
    def __call__(
        self,
        attn: CrossAttention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        # [B, C, H, W] -> [B, HW, C]
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        else:
            batch_size, sequence_length, _ = hidden_states.shape

        # choose where batch/seq come from (cross vs self)
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attn.group_norm is not None:
            # group_norm expects [B, C, L]
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        # projections
        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        # ---- head split: [B, S, C] -> [B, H, S, D] ----
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key   = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        # shapes now: [B, H, S, D]

        # optional Q/K norm (these modules expect [B, H, S, D])
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # ---- FlashAttention expects [B, S, H, D] ----
        q = query.transpose(1, 2).contiguous()   # [B, S, H, D]
        k = key.transpose(1, 2).contiguous()     # [B, S, H, D]
        v = value.transpose(1, 2).contiguous()   # [B, S, H, D]

        # # save & adjust dtype for FlashAttention (typically fp16/bf16 on GPU)
        # orig_dtype = q.dtype
        # if q.dtype not in (torch.float16, torch.bfloat16):
        #     q = q.to(torch.float16)
        #     k = k.to(torch.float16)
        #     v = v.to(torch.float16)

        # NOTE: attention_mask is ignored here; for FlashAttn masking you’d need
        # a different path (varlen or explicit mask handling).
        hidden_states = flash_attn_interface.flash_attn_func(
            q,
            k,
            v
        )
        # hidden_states: [B, S, H, D]

        # merge heads back: [B, S, H*D]
        hidden_states = hidden_states.reshape(batch_size, -1, attn.heads * head_dim)
        # hidden_states = hidden_states.to(orig_dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        # reshape back to image if needed
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        # residual + rescale
        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

@dataclass
class Transformer3DModelOutput(BaseOutput):
    sample: torch.FloatTensor


if is_xformers_available():
    import xformers
    import xformers.ops
else:
    xformers = None


class Transformer3DModel(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: Optional[int] = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        cross_attention_dim: Optional[int] = None,
        attention_bias: bool = False,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        use_linear_projection: bool = False,
        only_cross_attention: bool = False,
        upcast_attention: bool = False,
        use_motion_module: bool = False,
        unet_use_cross_frame_attention=None,
        unet_use_temporal_attention=None,
        add_audio_layer=False,
        audio_condition_method="cross_attn",
        custom_audio_layer: bool = False,
    ):
        super().__init__()
        self.use_linear_projection = use_linear_projection
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        inner_dim = num_attention_heads * attention_head_dim

        # Define input layers
        self.in_channels = in_channels

        self.norm = torch.nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=1e-6, affine=True)
        if use_linear_projection:
            self.proj_in = nn.Linear(in_channels, inner_dim)
        else:
            self.proj_in = nn.Conv2d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)

        if not custom_audio_layer:
            # Define transformers blocks
            self.transformer_blocks = nn.ModuleList(
                [
                    BasicTransformerBlock(
                        inner_dim,
                        num_attention_heads,
                        attention_head_dim,
                        dropout=dropout,
                        cross_attention_dim=cross_attention_dim,
                        activation_fn=activation_fn,
                        num_embeds_ada_norm=num_embeds_ada_norm,
                        attention_bias=attention_bias,
                        only_cross_attention=only_cross_attention,
                        upcast_attention=upcast_attention,
                        use_motion_module=use_motion_module,
                        unet_use_cross_frame_attention=unet_use_cross_frame_attention,
                        unet_use_temporal_attention=unet_use_temporal_attention,
                        add_audio_layer=add_audio_layer,
                        custom_audio_layer=custom_audio_layer,
                        audio_condition_method=audio_condition_method,
                    )
                    for d in range(num_layers)
                ]
            )
        else:
            self.transformer_blocks = nn.ModuleList(
                [
                    AudioTransformerBlock(
                        inner_dim,
                        num_attention_heads,
                        attention_head_dim,
                        dropout=dropout,
                        cross_attention_dim=cross_attention_dim,
                        activation_fn=activation_fn,
                        num_embeds_ada_norm=num_embeds_ada_norm,
                        attention_bias=attention_bias,
                        only_cross_attention=only_cross_attention,
                        upcast_attention=upcast_attention,
                        use_motion_module=use_motion_module,
                        unet_use_cross_frame_attention=unet_use_cross_frame_attention,
                        unet_use_temporal_attention=unet_use_temporal_attention,
                        add_audio_layer=add_audio_layer,
                    )
                    for d in range(num_layers)
                ]
            )

        # 4. Define output layers
        if use_linear_projection:
            self.proj_out = nn.Linear(in_channels, inner_dim)
        else:
            self.proj_out = nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0)

        if custom_audio_layer:
            self.proj_out = zero_module(self.proj_out)

    def forward(self, hidden_states, encoder_hidden_states=None, timestep=None, return_dict: bool = True):
        # Input
        assert hidden_states.dim() == 5, f"Expected hidden_states to have ndim=5, but got ndim={hidden_states.dim()}."
        video_length = hidden_states.shape[2]
        hidden_states = rearrange(hidden_states, "b c f h w -> (b f) c h w")

        # No need to do this for audio input, because different audio samples are independent
        # encoder_hidden_states = repeat(encoder_hidden_states, 'b n c -> (b f) n c', f=video_length)

        batch, channel, height, weight = hidden_states.shape
        residual = hidden_states

        hidden_states = self.norm(hidden_states)
        if not self.use_linear_projection:
            hidden_states = self.proj_in(hidden_states)
            inner_dim = hidden_states.shape[1]
            hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * weight, inner_dim)
        else:
            inner_dim = hidden_states.shape[1]
            hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * weight, inner_dim)
            hidden_states = self.proj_in(hidden_states)

        # Blocks
        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                video_length=video_length,
            )

        # Output
        if not self.use_linear_projection:
            hidden_states = hidden_states.reshape(batch, height, weight, inner_dim).permute(0, 3, 1, 2).contiguous()
            hidden_states = self.proj_out(hidden_states)
        else:
            hidden_states = self.proj_out(hidden_states)
            hidden_states = hidden_states.reshape(batch, height, weight, inner_dim).permute(0, 3, 1, 2).contiguous()

        output = hidden_states + residual

        output = rearrange(output, "(b f) c h w -> b c f h w", f=video_length)
        if not return_dict:
            return (output,)

        return Transformer3DModelOutput(sample=output)


class BasicTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        attention_bias: bool = False,
        only_cross_attention: bool = False,
        upcast_attention: bool = False,
        use_motion_module: bool = False,
        unet_use_cross_frame_attention=None,
        unet_use_temporal_attention=None,
        add_audio_layer=False,
        custom_audio_layer=False,
        audio_condition_method="cross_attn",
    ):
        super().__init__()
        self.only_cross_attention = only_cross_attention
        self.use_ada_layer_norm = num_embeds_ada_norm is not None
        self.unet_use_cross_frame_attention = unet_use_cross_frame_attention
        self.unet_use_temporal_attention = unet_use_temporal_attention
        self.use_motion_module = use_motion_module
        self.add_audio_layer = add_audio_layer

        # SC-Attn
        assert unet_use_cross_frame_attention is not None
        if unet_use_cross_frame_attention:
            raise NotImplementedError("SparseCausalAttention2D not implemented yet.")
        else:
            self.attn1 = CrossAttention(
                query_dim=dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                upcast_attention=upcast_attention,
                processor=FlashAttnProcessor2_0()
            )
        self.norm1 = AdaLayerNorm(dim, num_embeds_ada_norm) if self.use_ada_layer_norm else nn.LayerNorm(dim)

        # Cross-Attn
        if add_audio_layer and audio_condition_method == "cross_attn" and not custom_audio_layer:
            self.audio_cross_attn = AudioCrossAttn(
                dim=dim,
                cross_attention_dim=cross_attention_dim,
                num_attention_heads=num_attention_heads,
                attention_head_dim=attention_head_dim,
                dropout=dropout,
                attention_bias=attention_bias,
                upcast_attention=upcast_attention,
                num_embeds_ada_norm=num_embeds_ada_norm,
                use_ada_layer_norm=self.use_ada_layer_norm,
                zero_proj_out=False,
            )
        else:
            self.audio_cross_attn = None

        # Feed-forward
        self.ff = FeedForward(dim, dropout=dropout, activation_fn=activation_fn)
        self.norm3 = nn.LayerNorm(dim)

        # Temp-Attn
        assert unet_use_temporal_attention is not None
        if unet_use_temporal_attention:
            self.attn_temp = CrossAttention(
                query_dim=dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                upcast_attention=upcast_attention,
                processor=FlashAttnProcessor2_0()
            )
            nn.init.zeros_(self.attn_temp.to_out[0].weight.data)
            self.norm_temp = AdaLayerNorm(dim, num_embeds_ada_norm) if self.use_ada_layer_norm else nn.LayerNorm(dim)

    def set_use_memory_efficient_attention_xformers(self, use_memory_efficient_attention_xformers: bool):
        if not is_xformers_available():
            print("Here is how to install it")
            raise ModuleNotFoundError(
                "Refer to https://github.com/facebookresearch/xformers for more information on how to install"
                " xformers",
                name="xformers",
            )
        elif not torch.cuda.is_available():
            raise ValueError(
                "torch.cuda.is_available() should be True but is False. xformers' memory efficient attention is only"
                " available for GPU "
            )
        else:
            try:
                # Make sure we can run the memory efficient attention
                _ = xformers.ops.memory_efficient_attention(
                    torch.randn((1, 2, 40), device="cuda"),
                    torch.randn((1, 2, 40), device="cuda"),
                    torch.randn((1, 2, 40), device="cuda"),
                )
            except Exception as e:
                raise e
            self.attn1._use_memory_efficient_attention_xformers = use_memory_efficient_attention_xformers
            if self.audio_cross_attn is not None:
                self.audio_cross_attn.attn._use_memory_efficient_attention_xformers = (
                    use_memory_efficient_attention_xformers
                )
            # self.attn_temp._use_memory_efficient_attention_xformers = use_memory_efficient_attention_xformers

    def forward(
        self, hidden_states, encoder_hidden_states=None, timestep=None, attention_mask=None, video_length=None
    ):
        # SparseCausal-Attention
        norm_hidden_states = (
            self.norm1(hidden_states, timestep) if self.use_ada_layer_norm else self.norm1(hidden_states)
        )

        # if self.only_cross_attention:
        #     hidden_states = (
        #         self.attn1(norm_hidden_states, encoder_hidden_states, attention_mask=attention_mask) + hidden_states
        #     )
        # else:
        #     hidden_states = self.attn1(norm_hidden_states, attention_mask=attention_mask, video_length=video_length) + hidden_states

        # pdb.set_trace()
        if self.unet_use_cross_frame_attention:
            hidden_states = (
                self.attn1(norm_hidden_states, attention_mask=attention_mask, video_length=video_length)
                + hidden_states
            )
        else:
            hidden_states = self.attn1(norm_hidden_states, attention_mask=attention_mask) + hidden_states

        if self.audio_cross_attn is not None and encoder_hidden_states is not None:
            hidden_states = self.audio_cross_attn(
                hidden_states, encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask
            )

        # Feed-forward
        hidden_states = self.ff(self.norm3(hidden_states)) + hidden_states

        # Temporal-Attention
        if self.unet_use_temporal_attention:
            d = hidden_states.shape[1]
            hidden_states = rearrange(hidden_states, "(b f) d c -> (b d) f c", f=video_length)
            norm_hidden_states = (
                self.norm_temp(hidden_states, timestep) if self.use_ada_layer_norm else self.norm_temp(hidden_states)
            )
            hidden_states = self.attn_temp(norm_hidden_states) + hidden_states
            hidden_states = rearrange(hidden_states, "(b d) f c -> (b f) d c", d=d)

        return hidden_states


class AudioTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        attention_bias: bool = False,
        only_cross_attention: bool = False,
        upcast_attention: bool = False,
        use_motion_module: bool = False,
        unet_use_cross_frame_attention=None,
        unet_use_temporal_attention=None,
        add_audio_layer=False,
    ):
        super().__init__()
        self.only_cross_attention = only_cross_attention
        self.use_ada_layer_norm = num_embeds_ada_norm is not None
        self.unet_use_cross_frame_attention = unet_use_cross_frame_attention
        self.unet_use_temporal_attention = unet_use_temporal_attention
        self.use_motion_module = use_motion_module
        self.add_audio_layer = add_audio_layer

        # SC-Attn
        assert unet_use_cross_frame_attention is not None
        if unet_use_cross_frame_attention:
            raise NotImplementedError("SparseCausalAttention2D not implemented yet.")
        else:
            self.attn1 = CrossAttention(
                query_dim=dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                upcast_attention=upcast_attention,
                processor=FlashAttnProcessor2_0()
            )
        self.norm1 = AdaLayerNorm(dim, num_embeds_ada_norm) if self.use_ada_layer_norm else nn.LayerNorm(dim)

        self.audio_cross_attn = AudioCrossAttn(
            dim=dim,
            cross_attention_dim=cross_attention_dim,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            dropout=dropout,
            attention_bias=attention_bias,
            upcast_attention=upcast_attention,
            num_embeds_ada_norm=num_embeds_ada_norm,
            use_ada_layer_norm=self.use_ada_layer_norm,
            zero_proj_out=False,
        )

        # Feed-forward
        self.ff = FeedForward(dim, dropout=dropout, activation_fn=activation_fn)
        self.norm3 = nn.LayerNorm(dim)

    def set_use_memory_efficient_attention_xformers(self, use_memory_efficient_attention_xformers: bool):
        if not is_xformers_available():
            print("Here is how to install it")
            raise ModuleNotFoundError(
                "Refer to https://github.com/facebookresearch/xformers for more information on how to install"
                " xformers",
                name="xformers",
            )
        elif not torch.cuda.is_available():
            raise ValueError(
                "torch.cuda.is_available() should be True but is False. xformers' memory efficient attention is only"
                " available for GPU "
            )
        else:
            try:
                # Make sure we can run the memory efficient attention
                _ = xformers.ops.memory_efficient_attention(
                    torch.randn((1, 2, 40), device="cuda"),
                    torch.randn((1, 2, 40), device="cuda"),
                    torch.randn((1, 2, 40), device="cuda"),
                )
            except Exception as e:
                raise e
            self.attn1._use_memory_efficient_attention_xformers = use_memory_efficient_attention_xformers
            if self.audio_cross_attn is not None:
                self.audio_cross_attn.attn._use_memory_efficient_attention_xformers = (
                    use_memory_efficient_attention_xformers
                )
            # self.attn_temp._use_memory_efficient_attention_xformers = use_memory_efficient_attention_xformers

    def forward(
        self, hidden_states, encoder_hidden_states=None, timestep=None, attention_mask=None, video_length=None
    ):
        # SparseCausal-Attention
        norm_hidden_states = (
            self.norm1(hidden_states, timestep) if self.use_ada_layer_norm else self.norm1(hidden_states)
        )

        # pdb.set_trace()
        if self.unet_use_cross_frame_attention:
            hidden_states = (
                self.attn1(norm_hidden_states, attention_mask=attention_mask, video_length=video_length)
                + hidden_states
            )
        else:
            hidden_states = self.attn1(norm_hidden_states, attention_mask=attention_mask) + hidden_states

        if self.audio_cross_attn is not None and encoder_hidden_states is not None:
            hidden_states = self.audio_cross_attn(
                hidden_states, encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask
            )

        # Feed-forward
        hidden_states = self.ff(self.norm3(hidden_states)) + hidden_states

        return hidden_states


class AudioCrossAttn(nn.Module):
    def __init__(
        self,
        dim,
        cross_attention_dim,
        num_attention_heads,
        attention_head_dim,
        dropout,
        attention_bias,
        upcast_attention,
        num_embeds_ada_norm,
        use_ada_layer_norm,
        zero_proj_out=False,
    ):
        super().__init__()

        self.norm = AdaLayerNorm(dim, num_embeds_ada_norm) if use_ada_layer_norm else nn.LayerNorm(dim)
        self.attn = CrossAttention(
            query_dim=dim,
            cross_attention_dim=cross_attention_dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            upcast_attention=upcast_attention,
            processor=FlashAttnProcessor2_0()
        )

        if zero_proj_out:
            self.proj_out = zero_module(nn.Linear(dim, dim))

        self.zero_proj_out = zero_proj_out
        self.use_ada_layer_norm = use_ada_layer_norm

    def forward(self, hidden_states, encoder_hidden_states=None, timestep=None, attention_mask=None):
        previous_hidden_states = hidden_states
        hidden_states = self.norm(hidden_states, timestep) if self.use_ada_layer_norm else self.norm(hidden_states)

        if encoder_hidden_states.dim() == 4:
            encoder_hidden_states = rearrange(encoder_hidden_states, "b f n d -> (b f) n d")
            
        hidden_states = self.attn(
            hidden_states, encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask
        )

        if self.zero_proj_out:
            hidden_states = self.proj_out(hidden_states)
        return hidden_states + previous_hidden_states
