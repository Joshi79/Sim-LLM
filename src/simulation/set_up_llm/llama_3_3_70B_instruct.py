import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
import logging


logger = logging.getLogger(__name__)


def load_llama_model():
    model_id = "meta-llama/Llama-3.3-70B-Instruct"
    cache_dir = "/projects/2/managed_datasets/hf_cache_dir"
    cache_kwargs = dict(cache_dir=cache_dir, local_files_only=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        **cache_kwargs,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        **cache_kwargs,
        trust_remote_code=True
    )
    return model, tokenizer


class CachedLLMPipeline:
    def __init__(self, static_prompt, max_new_tokens= 250):
        """
        Initialize with a fully formatted static prompt.
        """
        self.model, self.tokenizer = load_llama_model()
        self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.max_new_tokens = max_new_tokens
        self.static_prompt = static_prompt.strip()

        # Tokenize the static prompt once
        self.static_tokens = self.tokenizer(
            self.static_prompt,
            return_tensors="pt",
            add_special_tokens=True
        )
        self.static_input_ids = self.static_tokens.input_ids.to(self.model.device)
        self.static_length = self.static_input_ids.shape[1]

        logger.info(f"Initialized with static prompt of {self.static_length} tokens")

    def invoke(self, dynamic_prompt):
        """
        Invoke the model with a dynamic prompt string and concatenats static prompt.
        """
        # Prepare the dynamic part
        dynamic_text = "\n" + dynamic_prompt.strip()

        dynamic_tokens = self.tokenizer(
            dynamic_text,
            return_tensors="pt",
            add_special_tokens=False
        )
        dynamic_input_ids = dynamic_tokens.input_ids.to(self.model.device)

        full_input_ids = torch.cat([
            self.static_input_ids,
            dynamic_input_ids
        ], dim=1)

        full_attention_mask = torch.ones_like(full_input_ids)


        with torch.no_grad():
            gen_ids = self.model.generate(
                input_ids=full_input_ids,
                attention_mask=full_attention_mask,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=False,
                temperature=0,
                top_p=0,
                use_cache=True,
            )

        # Extract only the newly generated tokens
        prompt_length = full_input_ids.shape[1]
        new_tokens = gen_ids[0, prompt_length:]
        raw_output = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        # Extract JSON block if present - fail save if output generates too much
        json_match = re.search(r"\{.*\}", raw_output, re.DOTALL)
        cleaned_output = json_match.group(0) if json_match else raw_output

        # Reconstruct full prompt for logging
        full_prompt = self.static_prompt + dynamic_text

        return cleaned_output, full_prompt, raw_output