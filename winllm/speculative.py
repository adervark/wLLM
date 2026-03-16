import torch
from typing import Optional, List
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from .engine import GenerationRequest, sample_token

class SpeculativeEngine:
    """Implements speculative decoding logic."""
    
    def __init__(
        self, 
        target_model: PreTrainedModel, 
        draft_model: PreTrainedModel, 
        tokenizer: PreTrainedTokenizerBase,
        num_speculative_tokens: int = 4
    ):
        self.target_model = target_model
        self.draft_model = draft_model
        self.tokenizer = tokenizer
        self.num_speculative_tokens = num_speculative_tokens
        self.device = next(target_model.parameters()).device

    @torch.inference_mode()
    def step(self, request: GenerationRequest) -> bool:
        """Perform one speculative decoding step for a single request.
        
        Returns True if the request should continue generating, False if it's finished.
        """
        # 1. Draft model generates N tokens
        draft_tokens = []
        current_past_key_values_draft = request._draft_past_key_values if hasattr(request, '_draft_past_key_values') else None
        
        # We need to maintain separate KV caches for draft and target
        # For simplicity in this implementation, we use the model's internal cache
        
        last_token = request.output_token_ids[-1]
        
        # Proposal loop
        proposal_input_ids = torch.tensor([[last_token]], device=self.device)
        temp_past_draft = request._draft_past_key_values
        
        for _ in range(self.num_speculative_tokens):
            outputs = self.draft_model(proposal_input_ids, past_key_values=temp_past_draft, use_cache=True)
            temp_past_draft = outputs.past_key_values
            
            next_logits = outputs.logits[:, -1, :]
            next_token_id = sample_token(next_logits, request.sampling_params, request.output_token_ids + draft_tokens)
            next_token = next_token_id.item()
            
            draft_tokens.append(next_token)
            proposal_input_ids = next_token_id.unsqueeze(0)
            
            if next_token == self.tokenizer.eos_token_id:
                break

        # 2. Target model verifies proposals in a single forward pass
        # Input to target: [last_token] + draft_tokens
        verify_input_ids = torch.tensor([[last_token] + draft_tokens], device=self.device)
        target_outputs = self.target_model(
            verify_input_ids, 
            past_key_values=request._past_key_values, 
            use_cache=True
        )
        
        request._past_key_values = target_outputs.past_key_values
        target_logits = target_outputs.logits[0, :, :] # (len, vocab)
        
        # 3. Acceptance evaluation
        accepted_count = 0
        for i in range(len(draft_tokens)):
            # Sample from target model at this position
            target_token_id = sample_token(target_logits[i:i+1, :], request.sampling_params, request.output_token_ids)
            target_token = target_token_id.item()
            
            if target_token == draft_tokens[i]:
                accepted_count += 1
                request.output_token_ids.append(target_token)
                if target_token == self.tokenizer.eos_token_id:
                    return False
            else:
                # Mismatch: append the target model's correction and stop verification
                request.output_token_ids.append(target_token)
                # We need to truncate the KV cache of the target model 
                # because we over-shot with the draft tokens
                # (This requires a more advanced KV cache manager that supports backtracking)
                break
        else:
            # All tokens accepted, append one more from the target model's next prediction
            last_target_token_id = sample_token(target_logits[-1:, :], request.sampling_params, request.output_token_ids)
            last_target_token = last_target_token_id.item()
            request.output_token_ids.append(last_target_token)
            if last_target_token == self.tokenizer.eos_token_id:
                return False

        # Update draft KV cache (simplified: just reset for now, or implement backtracking)
        request._draft_past_key_values = None 
        
        return True
