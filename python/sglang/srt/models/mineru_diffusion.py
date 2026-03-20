import logging
import re
from functools import partial
from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.managers.mm_utils import (
    MultiModalityDataPaddingPatternMultimodalTokens,
    embed_mm_inputs,
)
from sglang.srt.managers.schedule_batch import MultimodalDataItem, MultimodalInputs
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.qwen2_vl import (
    Qwen2VisionBlock,
    Qwen2VisionPatchEmbed,
    Qwen2VisionPatchMerger,
    Qwen2VisionRotaryEmbedding,
)
from sglang.srt.models.sdar import SDARForCausalLM
from sglang.srt.models.utils import compute_cu_seqlens_from_grid_numpy
from sglang.srt.utils import add_prefix, is_npu

logger = logging.getLogger(__name__)


class MinerUVisionModel(nn.Module):
    def __init__(
        self,
        vision_config: PretrainedConfig,
        norm_eps: float = 1e-6,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        patch_size = vision_config.patch_size
        temporal_patch_size = vision_config.temporal_patch_size
        spatial_merge_size = vision_config.spatial_merge_size
        in_chans = vision_config.in_chans
        embed_dim = vision_config.embed_dim
        depth = vision_config.depth
        num_heads = vision_config.num_heads
        mlp_ratio = vision_config.mlp_ratio

        self.spatial_merge_size = spatial_merge_size
        self.patch_embed = Qwen2VisionPatchEmbed(
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        head_dim = embed_dim // num_heads
        self.rotary_pos_emb = Qwen2VisionRotaryEmbedding(head_dim // 2)
        self.blocks = nn.ModuleList(
            [
                Qwen2VisionBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    norm_layer=norm_layer,
                    quant_config=quant_config,
                    prefix=add_prefix(f"blocks.{i}", prefix),
                )
                for i in range(depth)
            ]
        )

    @property
    def dtype(self) -> torch.dtype:
        return self.patch_embed.proj.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.blocks[0].mlp.fc2.weight.device

    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        pos_ids = []
        for i in range(grid_thw.size(0)):
            t, h, w = grid_thw[i].tolist()
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            hpos_ids = (
                hpos_ids.reshape(
                    h // self.spatial_merge_size,
                    self.spatial_merge_size,
                    w // self.spatial_merge_size,
                    self.spatial_merge_size,
                )
                .permute(0, 2, 1, 3)
                .flatten()
            )
            wpos_ids = (
                wpos_ids.reshape(
                    h // self.spatial_merge_size,
                    self.spatial_merge_size,
                    w // self.spatial_merge_size,
                    self.spatial_merge_size,
                )
                .permute(0, 2, 1, 3)
                .flatten()
            )
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb

    def forward(self, x: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        x = x.to(device=self.device, dtype=self.dtype)
        x = self.patch_embed(x)
        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())
        cu_seqlens = compute_cu_seqlens_from_grid_numpy(grid_thw)
        if is_npu():
            cu_seqlens = cu_seqlens.to("cpu")

        x = x.unsqueeze(1)
        for blk in self.blocks:
            x = blk(x, cu_seqlens=cu_seqlens, position_embeddings=position_embeddings)
        return x


class PerceiverProjection(nn.Module):
    def __init__(
        self,
        projection_type: str,
        in_dim: int,
        out_dim: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        pm_match = re.match(r"(?:patch_merger|pm)(\d+)x$", projection_type)
        if pm_match is None:
            raise ValueError(
                f"Only patch_merger-style projectors are supported, got: {projection_type}"
            )
        merge_size = int(pm_match.group(1))
        self.projection = Qwen2VisionPatchMerger(
            d_model=out_dim,
            context_dim=in_dim,
            spatial_merge_size=merge_size,
            quant_config=quant_config,
            prefix=prefix,
        )

    def forward(self, input_embeds: torch.Tensor) -> torch.Tensor:
        return self.projection(input_embeds)


class MinerUDiffusionForConditionalGeneration(nn.Module):
    merge_by_field_config = True

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        **kwargs,
    ) -> None:
        super().__init__()
        self.config = config
        self.quant_config = quant_config

        if config.vision_model_type != "qwen2_vl":
            raise ValueError(
                f"Only qwen2_vl vision towers are supported, got: {config.vision_model_type}"
            )

        self.vision_model = MinerUVisionModel(
            config.vision_model_config,
            norm_eps=getattr(config, "rms_norm_eps", 1e-6),
            quant_config=quant_config,
            prefix=add_prefix("vision_model", prefix),
        )
        self.vision_abstractor = PerceiverProjection(
            projection_type=config.vision_projector_type,
            in_dim=config.vision_model_config.embed_dim,
            out_dim=config.language_model_config.hidden_size,
            quant_config=quant_config,
        )
        self.language_model = SDARForCausalLM(
            config=config.language_model_config,
            quant_config=quant_config,
            prefix=add_prefix("language_model", prefix),
        )

    def get_input_embeddings(self):
        return self.language_model.model.embed_tokens

    def get_language_model(self) -> torch.nn.Module:
        return self.language_model

    def get_embed_and_head(self):
        return (
            self.language_model.model.embed_tokens.weight,
            self.language_model.lm_head.weight,
        )

    def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):
        pattern = MultiModalityDataPaddingPatternMultimodalTokens()
        return pattern.pad_input_tokens(input_ids, mm_inputs)

    def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        pixel_values = torch.cat([item.feature for item in items], dim=0).type(
            self.vision_model.dtype
        )
        image_grid_thw = torch.concat([item.image_grid_thw for item in items], dim=0)
        assert pixel_values.dim() == 2, pixel_values.dim()
        assert image_grid_thw.dim() == 2, image_grid_thw.dim()
        image_features = self.vision_model(pixel_values, image_grid_thw)
        return self.vision_abstractor(image_features)

    def _should_apply_mm_embeds(
        self, input_ids: torch.Tensor, forward_batch: ForwardBatch
    ) -> bool:
        if (
            forward_batch.forward_mode.is_decode()
            or forward_batch.forward_mode.is_target_verify()
            or not forward_batch.contains_mm_inputs()
        ):
            return False

        placeholder_values = [
            item.pad_value
            for mm_input in (forward_batch.mm_inputs or [])
            if mm_input is not None
            for item in mm_input.mm_items
            if item is not None
        ]
        if not placeholder_values:
            return False

        return torch.isin(
            input_ids,
            torch.as_tensor(placeholder_values, device=input_ids.device),
        ).any().item()

    def prepare_cuda_graph_input_embeds(
        self,
        input_ids: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor,
    ) -> torch.Tensor:
        embed_tokens = self.get_input_embeddings()
        if self._should_apply_mm_embeds(input_ids, forward_batch):
            mm_inputs_list = [
                mm_input for mm_input in forward_batch.mm_inputs if mm_input is not None
            ]
            extend_prefix_lens = [
                prefix_len
                for i, prefix_len in enumerate(forward_batch.extend_prefix_lens_cpu)
                if forward_batch.mm_inputs[i] is not None
            ]
            extend_seq_lens = [
                seq_len
                for i, seq_len in enumerate(forward_batch.extend_seq_lens_cpu)
                if forward_batch.mm_inputs[i] is not None
            ]
            computed_input_embeds, _ = embed_mm_inputs(
                mm_inputs_list=mm_inputs_list,
                extend_prefix_lens=extend_prefix_lens,
                extend_seq_lens=extend_seq_lens,
                input_ids=input_ids,
                input_embedding=embed_tokens,
                multimodal_model=self,
            )
        else:
            computed_input_embeds = embed_tokens(input_ids)

        input_embeds.copy_(computed_input_embeds.to(dtype=input_embeds.dtype))
        return input_embeds

    def _build_input_embeds(
        self,
        input_ids: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        embed_tokens = self.get_input_embeddings()
        if self._should_apply_mm_embeds(input_ids, forward_batch):
            mm_inputs_list = [
                mm_input for mm_input in forward_batch.mm_inputs if mm_input is not None
            ]
            extend_prefix_lens = [
                prefix_len
                for i, prefix_len in enumerate(forward_batch.extend_prefix_lens_cpu)
                if forward_batch.mm_inputs[i] is not None
            ]
            extend_seq_lens = [
                seq_len
                for i, seq_len in enumerate(forward_batch.extend_seq_lens_cpu)
                if forward_batch.mm_inputs[i] is not None
            ]
            input_embeds, _ = embed_mm_inputs(
                mm_inputs_list=mm_inputs_list,
                extend_prefix_lens=extend_prefix_lens,
                extend_seq_lens=extend_seq_lens,
                input_ids=input_ids,
                input_embedding=embed_tokens,
                multimodal_model=self,
            )
            forward_batch.mm_inputs = None
            forward_batch.mm_input_embeds = input_embeds
        else:
            input_embeds = embed_tokens(input_ids)

        if forward_batch.input_embeds is not None:
            forward_batch.input_embeds.copy_(input_embeds)
            input_embeds = forward_batch.input_embeds
        return input_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds=None,
        get_embedding: bool = False,
    ):
        positions = positions.to(torch.int32)
        if input_embeds is None:
            input_embeds = self._build_input_embeds(input_ids, forward_batch)
        original_positions = forward_batch.positions
        original_out_cache_loc = forward_batch.out_cache_loc
        forward_batch.positions = positions
        if (
            forward_batch.out_cache_loc is not None
            and forward_batch.out_cache_loc.dtype != positions.dtype
        ):
            forward_batch.out_cache_loc = forward_batch.out_cache_loc.to(positions.dtype)
        try:
            return self.language_model(
                input_ids=input_ids,
                positions=positions,
                forward_batch=forward_batch,
                input_embeds=input_embeds,
            )
        finally:
            forward_batch.positions = original_positions
            forward_batch.out_cache_loc = original_out_cache_loc

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        language_model_weights: list[Tuple[str, torch.Tensor]] = []

        for name, loaded_weight in weights:
            if name.startswith("language_model."):
                language_model_weights.append(
                    (name[len("language_model.") :], loaded_weight)
                )
                continue

            if name.startswith("vision_model."):
                name = name.replace(".attn.qkv.", ".attn.qkv_proj.")

            if name.endswith(".bias") and name not in params_dict:
                continue

            param = params_dict.get(name)
            if param is None:
                logger.warning("Parameter %s not found in params_dict", name)
                continue
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)

        if language_model_weights:
            self.language_model.load_weights(language_model_weights)


EntryClass = [MinerUDiffusionForConditionalGeneration]
