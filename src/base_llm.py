from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

DEFAULT_MODEL = "sshleifer/tiny-gpt2"

class DraftLM:
    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def next_logits(self, input_ids: torch.Tensor):
        out = self.model(input_ids.to(self.device))
        return out.logits[:, -1, :]  # [1, vocab]

    def tokenize(self, text: str):
        return self.tokenizer(text, return_tensors="pt")

    def decode(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    @property
    def vocab_size(self) -> int:
        return self.model.config.vocab_size

