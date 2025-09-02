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
    '''
    reeturns a reverse complement for a sequence in 5' to 3' orientation
    '''
    output=""
    bases={"A":"T", "C":"G", "G":"C", "T":"A"}
    for i in seq:
        output+=bases[i]

    return output[::-1]


def remove_gaps(gapp):
    '''
    removes gaps from a sequence in the alignmnet
    '''
    output=""
    for i in gapp:
        if i != "-":
            output+=i

    return output
    
def parse_genbank_to_dataframe(filepath):
    """
    Parses a GenBank file to extract gene information and returns a pandas DataFrame.

    :param filepath: str
        The file path to the GenBank file.
    :return: pd.DataFrame
        A DataFrame containing gene information.
    """
    gene_data = {'Gene': [], 'Length': [], 'Strand': [], 'Type': [], 'Number': [],'Start': [], 'End': []}
    with open(filepath, 'r') as file:
        genome_record = SeqIO.read(file, 'genbank')
        
        for i, feature in enumerate(genome_record.features):
             if (feature.type!="gene") & (feature.type!="source"):
                try:
                    gene_name = feature.qualifiers.get('gene', [f'Unk_{i}'])[0]
                    
                except KeyError:
                    gene_name = np.nan
                try:
                    db_xref = feature.qualifiers["db_xref"]
                    gene_number=[xref.split(":")[1] for xref in db_xref if xref.startswith("UniProtKB/Swiss-Prot:")]
                    if len(gene_number)==1:
                        gene_number=gene_number[0]
                    else:
                        gene_number= np.nan
                except KeyError:
                    gene_number= np.nan
                    
                start, end = feature.location.start, feature.location.end
                gene_length = end - start
                strand = feature.location.strand
        
                gene_data['Gene'].append(gene_name)
                gene_data['Number'].append(gene_number)
                gene_data['Length'].append(gene_length)
                gene_data['Strand'].append(strand)
                gene_data['Start'].append(start)
                gene_data['End'].append(end)
                gene_data['Type'].append(feature.type)
            
    genome_DF = pd.DataFrame(gene_data)
    genome_DF['Strand'] = genome_DF['Strand'].map({1: 'FW', -1: 'RC'})
    return genome_DF
