import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from transformers import LlavaNextForConditionalGeneration
from transformers import LlavaNextPreTrainedModel, LlavaNextConfig
from safetensors.torch import load_file
import os

try:
    from transformers import LlavaNextProcessor

    def load_processor(model_id: str):
        return LlavaNextProcessor.from_pretrained(model_id)
except ImportError:
    from transformers import AutoProcessor

    def load_processor(model_id: str):
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        if not hasattr(processor, "tokenizer"):
            raise ValueError("Loaded processor lacks tokenizer attribute; please update transformers.")
        return processor

if torch.cuda.is_available():
    device = torch.device("cuda")
    # device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.device_count() > args.gpu_id else "cuda:0")
else:
    device = torch.device("cpu")


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ValueModel(LlavaNextPreTrainedModel):
    def __init__(self, base_model_pth, hidden_size=None, output_size=1, shared_base=None):
        if shared_base is not None:
            config = shared_base.config
        else:
            config = LlavaNextConfig.from_pretrained(base_model_pth)
        super(ValueModel, self).__init__(config)
        self.base_model_pth = base_model_pth
        self.processor = load_processor(self.base_model_pth)
        self.flatten = Flatten()
        if shared_base is not None:
            self._llava_wrapper = shared_base
        else:
            base_model = LlavaNextForConditionalGeneration.from_pretrained(
                self.base_model_pth,
                low_cpu_mem_usage=True,
                torch_dtype=torch.float16,
                local_files_only=True
                # device_map=device
            )
            self._llava_wrapper = base_model
        self.llava_encoder = getattr(self._llava_wrapper, "model", None)
        if self.llava_encoder is None:
            self.llava_encoder = self._llava_wrapper
        self.hidden_size = None
        self.output_size = output_size
        if hidden_size is not None:
            self._build_head(hidden_size)
        else:
            self.v_head = None

    def forward(self, inputs, attention_mask=None):
        # print(inputs.keys())
        if self.v_head is None or self.hidden_size is None:
            raise RuntimeError("Value head not initialized; call from_pretrained first.")
        # latent_output = self.llava_encoder(**inputs, output_hidden_states=True, max_embed_length=3000)
        latent_output = self.llava_encoder(**inputs, output_hidden_states=True)
        hidden_states = self.flatten(latent_output.hidden_states[-1])
        if hidden_states.size(-1) != self.hidden_size:
            if hidden_states.size(-1) < self.hidden_size:
                pad_width = self.hidden_size - hidden_states.size(-1)
                hidden_states = F.pad(hidden_states, (0, pad_width))
            else:
                hidden_states = hidden_states[..., :self.hidden_size]
        output = self.v_head(hidden_states)
        return output

    def _build_head(self, in_dim):
        self.hidden_size = in_dim
        self.v_head = nn.Linear(in_dim, self.output_size)
        self._initialize_weights(self.v_head)

    def _initialize_weights(self, module):
        for layer in module.modules():
            if isinstance(layer, nn.Linear):
                init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    init.zeros_(layer.bias)

    def from_pretrained(self, pretrained_model_pth, load_backbone: bool = True):
        complete_state_dict = {}

        # Load all 4 safetensor shards
        for i in range(4):
            complete_state_dict.update(
                load_file(
                    os.path.join(
                        pretrained_model_pth,
                        "model-0000{}-of-00004.safetensors".format(i + 1)
                    )
                )
            )

        # Apply both replacements: remove 'llava_encoder.' and fix 'language_model.model.'
        llava_encoder_dict = {
            k.replace('llava_encoder.', '')
            .replace('language_model.model.', 'language_model.'): v
            for k, v in complete_state_dict.items()
        }

        # Separate out value head weights
        value_keys = ["v_head.0.weight", "v_head.0.bias"]
        head_weight = complete_state_dict.get("v_head.0.weight")
        head_bias = complete_state_dict.get("v_head.0.bias")

        for key in value_keys:
            if key in complete_state_dict:
                del llava_encoder_dict[key]

        # Load encoder weights
        if load_backbone:
            self.llava_encoder.load_state_dict(llava_encoder_dict, strict=False)

        if head_weight is not None:
            in_dim = head_weight.shape[1]
            if self.hidden_size is None:
                self._build_head(in_dim)
            elif self.hidden_size != in_dim:
                self._build_head(in_dim)

            state = {}
            state["weight"] = head_weight
            if head_bias is not None:
                state["bias"] = head_bias
            self.v_head.load_state_dict(state, strict=True)
        elif self.hidden_size is None:
            raise ValueError("Value head weights missing and hidden_size not provided; cannot initialize value head.")

    def to_value_device(self, device=None, dtype=None):
        if self.v_head is not None and (device is not None or dtype is not None):
            self.v_head.to(device=device, dtype=dtype)
        elif self.v_head is not None and device is None and dtype is None:
            pass
        if device is not None:
            self.flatten.to(device=device)
        return self
