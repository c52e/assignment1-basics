from cs336_basics import *

class TextGenerator(object):
    def __init__(self, model: torch.nn.Module, tokenizer: Tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        eos_token = tokenizer.special_tokens[0]
        assert eos_token == '<|endoftext|>', "The first special token must be the end-of-text token '<|endoftext|>'"
        self.eos_token_id = self.tokenizer.encode(eos_token)[0]
        
    def generate(self, prompt: str, max_length: int, temperature: float = 1.0, top_k: int = 0) -> str:
        self.model.eval()
        device = next(self.model.parameters()).device
        input_ids_list = self.tokenizer.encode(prompt)
        if len(input_ids_list) > self.model.context_length:
            input_ids_list = input_ids_list[-self.model.context_length:]
        input_ids = torch.tensor([input_ids_list], dtype=torch.int32, device=device)
        with torch.no_grad():
            for _ in range(max_length - len(input_ids_list)):
                curr_seq_len = input_ids.size(1)
                if curr_seq_len >= self.model.context_length:
                    break
                logits = self.model(input_ids)
                next_token_logits = logits[0, -1, :] / (temperature + 1e-9)
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][-1]
                    next_token_logits[indices_to_remove] = float('-inf')
                probs = softmax(next_token_logits, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1)
                input_ids = torch.cat([input_ids, next_token_id.unsqueeze(0)], dim=1)
                if next_token_id.item() == self.eos_token_id:
                    break
        output_ids = input_ids[0].cpu().tolist()
        return self.tokenizer.decode(output_ids)
