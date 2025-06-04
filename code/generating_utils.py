from evo.scoring import prepare_batch
from scipy import special
from scipy.special import softmax, log_softmax
import random
import torch
import numpy as np

def logits_to_logprobs(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    trim_bos: bool = True,
) -> torch.Tensor:
    """
    Takes in a tensor of logits of dimension (batch, length, vocab).
    Computes the log-likelihoods using a softmax along the vocab dimension.
    Uses the `input_ids` to index into the log-likelihoods and returns the likelihood
    of the provided sequence at each position with dimension (batch, length).
    """
    softmax_logprobs = torch.log_softmax(logits, dim=-1)
    if trim_bos:
        softmax_logprobs = softmax_logprobs[:, :-1] # Remove last prediction.
        input_ids = input_ids[:, 1:] # Trim BOS added by tokenizer.
    assert(softmax_logprobs.shape[1] == input_ids.shape[1])

    logprobs = torch.gather(
        softmax_logprobs,       # Gather likelihoods...
        2,                      # along the vocab dimension...
        input_ids.unsqueeze(-1) # using the token ids to index.
    ).squeeze(-1)

    return logprobs

def perposition_scores(seqs,
    model,
    tokenizer,
    device='cuda:0',
                      ):
    input_ids, seq_lengths = prepare_batch(seqs, tokenizer, device=device, prepend_bos=True)
    assert(len(seq_lengths) == input_ids.shape[0])

    with torch.inference_mode():
        logits, _ = model(input_ids) # (batch, length, vocab)

    logprobs = logits_to_logprobs(logits, input_ids, trim_bos=True)
    logprobs = logprobs.float().cpu().numpy()
    
    sequence_scores = [
        logprobs[idx][:seq_lengths[idx]] for idx in range(len(seq_lengths))
    ]
    output=[]
    for i in sequence_scores:
        output.append([math.exp(j) for j in i])
    return output


def complement_5_strand(seq):
    output=""
    bases={"A":"T", "C":"G", "G":"C", "T":"A"}
    for i in seq:
        output+=bases[i]

    return output[::-1]
