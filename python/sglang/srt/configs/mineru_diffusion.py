from typing import Union

import torch
from transformers import AutoProcessor, CONFIG_MAPPING, PretrainedConfig
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import load_image
from transformers.models.qwen2_vl.configuration_qwen2_vl import Qwen2VLVisionConfig
from transformers.processing_utils import ProcessorMixin

from sglang.srt.multimodal.customized_mm_processor_utils import (
    register_customized_processor,
)


class MinerUDiffusionProcessor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]
    valid_kwargs = ["chat_template"]
    optional_attributes = ["chat_template"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = ("Qwen2Tokenizer", "Qwen2TokenizerFast")

    special_tokens = ["<|image_pad|>", "<|vision_start|>", "<|vision_end|>"]

    def __init__(self, image_processor=None, tokenizer=None, chat_template=None, **kwargs):
        if chat_template == "auto" and tokenizer is not None:
            chat_template = tokenizer.chat_template
        super().__init__(image_processor, tokenizer, chat_template=chat_template)
        self.downsample_size = kwargs.pop("downsample_size", 2)
        self.image_token = kwargs.pop(
            "image_token",
            getattr(tokenizer, "image_token", "<|image_pad|>"),
        )
        self.special_tokens = kwargs.pop(
            "special_tokens",
            ["<|image_pad|>", "<|vision_start|>", "<|vision_end|>"],
        )

        try:
            self.tokenizer.add_special_tokens(
                {"additional_special_tokens": self.special_tokens},
                replace_additional_special_tokens=False,
            )
        except TypeError:
            self.tokenizer.add_special_tokens(
                {"additional_special_tokens": self.special_tokens}
            )
        self.image_token_id = (
            tokenizer.image_token_id
            if getattr(tokenizer, "image_token_id", None) is not None
            else tokenizer.convert_tokens_to_ids(self.image_token)
        )

    def _process_images(self, images, **kwargs) -> dict[str, torch.Tensor]:
        loaded_images = [load_image(image) for image in images]
        if loaded_images:
            return self.image_processor(
                images=loaded_images,
                return_tensors="pt",
                **kwargs,
            )

        patch_size = getattr(self.image_processor, "patch_size", 14)
        return {
            "pixel_values": torch.zeros(
                (0, 3 * 2 * (patch_size**2)), dtype=torch.float32
            ),
            "image_grid_thw": torch.zeros((0, 3), dtype=torch.int64),
        }

    def _expand_image_tokens(
        self, text: list[str], image_grid_thw: torch.Tensor
    ) -> list[str]:
        expanded_text = text.copy()
        image_token_lengths = (
            image_grid_thw.prod(dim=1) // (self.downsample_size**2)
        ).tolist()
        image_index = 0
        for index in range(len(expanded_text)):
            while self.image_token in expanded_text[index]:
                if image_index >= len(image_token_lengths):
                    raise ValueError(
                        "Wrong image token count, more image tokens than processed images."
                    )
                expanded_text[index] = expanded_text[index].replace(
                    self.image_token,
                    "<|placeholder|>" * image_token_lengths[image_index],
                    1,
                )
                image_index += 1
            expanded_text[index] = expanded_text[index].replace(
                "<|placeholder|>", self.image_token
            )
        if image_index != len(image_token_lengths):
            raise ValueError(
                "Wrong image token count, "
                f"image_token_count({image_index}) != image_count({len(image_token_lengths)})"
            )
        return expanded_text

    @staticmethod
    def _count_image_embeds(image_grid_thw: torch.Tensor, downsample_size: int) -> int:
        return int((image_grid_thw.prod(dim=1) // (downsample_size**2)).sum().item())

    def _validate_image_inputs(self, input_ids, image_grid_thw: torch.Tensor) -> None:
        if isinstance(input_ids, torch.Tensor):
            image_token_count = torch.count_nonzero(
                input_ids == self.image_token_id
            ).item()
        else:
            image_token_count = sum(row.count(self.image_token_id) for row in input_ids)
        image_embed_count = self._count_image_embeds(
            image_grid_thw, self.downsample_size
        )
        if image_token_count != image_embed_count:
            raise ValueError(
                "Wrong image embed token count, "
                f"image_embed_token_count({image_token_count}) != image_embed_count({image_embed_count})"
            )

    def __call__(self, images=None, text=None, **kwargs) -> BatchFeature:
        if images is None:
            images = []
        return_tensors = kwargs.pop("return_tensors", None)
        image_kwargs = {}
        if "device" in kwargs:
            image_kwargs["device"] = kwargs.pop("device")
        image_inputs = self._process_images(images, **image_kwargs)
        if text is None:
            return BatchFeature(data=image_inputs, tensor_type=return_tensors)

        if not isinstance(text, list):
            text = [text]
        text = self._expand_image_tokens(text.copy(), image_inputs["image_grid_thw"])
        text_inputs = self.tokenizer(text, return_tensors=return_tensors, **kwargs)
        self._validate_image_inputs(
            text_inputs["input_ids"], image_inputs["image_grid_thw"]
        )
        return BatchFeature(
            data={**text_inputs, **image_inputs}, tensor_type=return_tensors
        )

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    def post_process_image_text_to_text(
        self,
        generated_outputs: Union[list[list[int]], torch.Tensor],
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = False,
        **kwargs,
    ):
        return self.tokenizer.batch_decode(
            generated_outputs,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs,
        )

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(
            dict.fromkeys(tokenizer_input_names + image_processor_input_names)
        )


class SDARConfig(PretrainedConfig):
    model_type = "sdar"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=151936,
        hidden_size=2048,
        intermediate_size=6144,
        num_hidden_layers=28,
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=128,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=1000000.0,
        rope_scaling=None,
        attention_bias=False,
        use_sliding_window=False,
        sliding_window=None,
        max_window_layers=28,
        attention_dropout=0.0,
        block_size=32,
        enable_block_generation=True,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window
        self.max_window_layers = max_window_layers
        self.num_key_value_heads = (
            num_attention_heads if num_key_value_heads is None else num_key_value_heads
        )
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.block_size = block_size
        self.enable_block_generation = enable_block_generation
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


@register_customized_processor(processor_class=MinerUDiffusionProcessor)
class MinerUDiffusionConfig(PretrainedConfig):
    model_type = "mineru_diffusion"
    sub_configs = {"vision_config": Qwen2VLVisionConfig, "text_config": SDARConfig}
    keys_to_ignore_at_inference = ["past_key_values"]
    architectures = ["MinerUDiffusionForConditionalGeneration"]

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        language_model_config=None,
        vision_model_config=None,
        image_token_id=151655,
        video_token_id=151656,
        vision_start_token_id=151652,
        vision_end_token_id=151653,
        mask_token_id=151669,
        image_size=512,
        patch_size=16,
        downsample_ratio=0.5,
        vision_projector_type="patch_merger2x",
        vision_select_layer=-2,
        tie_word_embeddings=False,
        **kwargs,
    ):
        kwargs.pop("rm_vit_merger", None)
        top_level_torch_dtype = kwargs.pop("torch_dtype", None)
        if text_config is None:
            text_config = language_model_config
        if vision_config is None:
            vision_config = vision_model_config

        if isinstance(text_config, dict):
            self.text_config = SDARConfig(**text_config)
        elif text_config is None:
            self.text_config = SDARConfig()
        else:
            self.text_config = text_config

        if isinstance(vision_config, dict):
            vision_model_type = vision_config.get("model_type", "")
            if vision_model_type != "qwen2_vl":
                raise ValueError(
                    f"Unsupported vision config type: {vision_model_type}"
                )
            self.vision_config = Qwen2VLVisionConfig(**vision_config)
        elif vision_config is None:
            self.vision_config = Qwen2VLVisionConfig()
        else:
            self.vision_config = vision_config

        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id
        self.mask_token_id = mask_token_id
        self.image_size = image_size
        self.patch_size = patch_size
        self.downsample_ratio = downsample_ratio
        self.vision_projector_type = vision_projector_type
        self.vision_select_layer = vision_select_layer
        self.tie_word_embeddings = tie_word_embeddings
        self.auto_map = {
            "AutoConfig": "configuration_mineru_diffusion.MinerUDiffusionConfig",
            "AutoModel": "modeling_mineru_diffusion.MinerUDiffusionForConditionalGeneration",
            "AutoModelForCausalLM": "modeling_mineru_diffusion.MinerUDiffusionForConditionalGeneration",
            "AutoProcessor": "processing_mineru_diffusion.MinerUDiffusionProcessor",
        }

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            bos_token_id=getattr(self.text_config, "bos_token_id", None),
            eos_token_id=getattr(self.text_config, "eos_token_id", None),
            pad_token_id=getattr(self.text_config, "pad_token_id", None),
            **kwargs,
        )
        self.torch_dtype = getattr(
            self.text_config, "torch_dtype", top_level_torch_dtype
        )

    @property
    def language_model_config(self):
        return self.text_config

    @property
    def vision_model_config(self):
        return self.vision_config

    @property
    def vision_model_type(self):
        return getattr(self.vision_config, "model_type", None)

    @property
    def hidden_size(self):
        return self.text_config.hidden_size


for name, cls in {
    "sdar": SDARConfig,
    "mineru_diffusion": MinerUDiffusionConfig,
}.items():
    try:
        CONFIG_MAPPING.register(name, cls)
    except Exception:
        CONFIG_MAPPING._extra_content[name] = cls

AutoProcessor.register(MinerUDiffusionConfig, MinerUDiffusionProcessor, exist_ok=True)
