import torch
import random
from typing import Literal
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))

from datasets.features.video import Example
from transformers import RobertaTokenizerFast, PretrainedConfig
from dataclasses import dataclass, asdict

from datasets import load_dataset
from pprintpp import pprint

@dataclass
class RobertaConfig(PretrainedConfig):
    ### Tokenizer Config
    vocab_size: int = 50265
    start_token: int = 0
    end_token: int = 2
    pad_token: int = 2
    mask_token: int = 50264

    ### Transformer Config ###
    embedding_dimension: int = 768
    num_transformer_blocks: int = 12
    num_attention_heads: int = 12
    mlp_ratio: int = 4
    layer_norm_eps: float = 1e-6
    hidden_dropout_p: float = 0.1
    attention_dropout_p: float = 0.1
    context_length: int = 512

    ### Masking Config ###
    masking_prob: float = 0.15

    ### Huggingface Config ###
    hf_model_name: str = "deepset/roberta-base-squad2"

    ### Model Config ###
    pretrained_backbone: Literal["pretrained", "pretrained_huggingface", "random"] = "pretrained"
    max_position_embeddings: int = 512
    path_to_pretrained_weights: str = None

    ### Added in to_dict() method so this Config is compatible with Huggingface Trainer!!! ###
    def to_dict(self):
        return asdict(self)


def random_masking_text(tokens,
                        special_tokens_mask,
                        vocab_size=50264,
                        special_ids=(0, 1, 2, 3, 50264),
                        mask_ratio=0.15,
                        mask_token=50264):
    """
    Function for our random masking of tokens (excluding special tokens). This follow the logic provided
    by BERT/RoBERTa:

        - Select 15% of the tokens for masking
            - 80% of the selected tokens are replaced with a mask token
            - 10% of the selected tokens are replaced with another random token
            - 10% of the selected tokens are left alone

    This is almost identical to the masking function in our introductory jupyter notebook walkthrough of
    masked language modeling, but some minor changes are made to apply masking to batches of tokens
    rather than just one sequence at a time!
    """

    ### Create Random Uniform Sample Tensor ###
    random_masking = torch.rand(*tokens.shape)

    ### Set Value of Special Tokens to 1 so we DONT MASK THEM ###
    random_masking[special_tokens_mask == 1] = 1

    ### Get Boolean of Words under Masking Threshold ###
    random_masking = (random_masking < mask_ratio)

    ### Create Labels ###
    labels = torch.full((tokens.shape), -100)
    labels[random_masking] = tokens[random_masking]

    ### Get Indexes of True ###
    random_selected_idx = random_masking.nonzero()

    ### 80% Of the Time Replace with Mask Token ###
    masking_flag = torch.rand(len(random_selected_idx))
    masking_flag = (masking_flag < 0.8)
    selected_idx_for_masking = random_selected_idx[masking_flag]

    ### Seperate out remaining indexes to be assigned ###
    unselected_idx_for_masking = random_selected_idx[~masking_flag]

    ### 10% of the time (or 50 percent of the remaining 20%) we fill with random token ###
    ### The remaining times, leave the text as is ###
    masking_flag = torch.rand(len(unselected_idx_for_masking))
    masking_flag = (masking_flag < 0.5)
    selected_idx_for_random_filling = unselected_idx_for_masking[masking_flag]
    selected_idx_to_be_left_alone = unselected_idx_for_masking[~masking_flag]

    ### Fill Mask Tokens ###
    if len(selected_idx_for_masking) > 0:
        tokens[selected_idx_for_masking[:, 0], selected_idx_for_masking[:, 1]] = mask_token

    ### Fill Random Tokens ###
    if len(selected_idx_for_random_filling) > 0:
        non_special_ids = list(set(range(vocab_size)) - set(special_ids))
        randomly_selected_tokens = torch.tensor(random.sample(non_special_ids, len(selected_idx_for_random_filling)))
        tokens[selected_idx_for_random_filling[:, 0], selected_idx_for_random_filling[:, 1]] = randomly_selected_tokens

    return tokens, labels



def ExtractiveQAPreProcesing():

    tokenizer = RobertaTokenizerFast.from_pretrained("deepset/roberta-base-squad2")

    def char2token_mapping(examples):
        #
        # pprint(examples)

        questions = [q.strip() for sublist in examples["questions"] for q in sublist]
        # pprint(questions)
        stories = []
        for idx, sublist in enumerate(examples["questions"]):
            stories.extend([examples["story"][idx]] * len(sublist))

        # Now both questions and stories are 1D lists of the same length
        input = tokenizer(
            text=questions,
            text_pair=stories,
            max_length=512,
            truncation="only_second",
            return_offsets_mapping=True,
            padding="max_length",
        )
        offset_mapping = input.pop("offset_mapping")
        # pprint(input)
        answers = examples["answers"]
        input_text =[]
        answer_start = []
        answer_end=[]

        for key in answers:
            input_text.extend(key["input_text"])
            answer_start.extend(key["answer_start"])
            answer_end.extend(key["answer_end"])


        starting_token_idxs = []
        ending_token_idxs = []
        convert_data = {}

        for i, offset in enumerate(offset_mapping):

            start_char = answer_start[i]
            end_char = answer_end[i]

            # if start_char == -1 or end_char == -1:
            #     starting_token_idxs.append(0)
            #     ending_token_idxs.append(0)
            #     continue

            sequencen_ids = input.sequence_ids(i)

            context_start = None
            context_end = None

            for idx, id in enumerate(sequencen_ids):
                if context_start is None and id == 1:
                    context_start = idx
                elif context_start is not None and id != 1:
                    context_end = idx - 1
                    break
                elif context_start is not None and idx == len(sequencen_ids) - 1:
                    context_end = idx


            context_start_char = offset[context_start][0]
            context_end_char = offset[context_end][-1]

            if (start_char >= context_start_char) and (end_char <= context_end_char):
                # print(start_char, end_char)
                start_token_idx = None
                end_token_idx = None
                for token_idx, (offsets, seq_id) in enumerate(zip(offset, sequencen_ids)):
                    if seq_id == 1:
                        if start_char in range(offsets[0], offsets[1] + 1):
                            start_token_idx = token_idx
                        if end_char in range(offsets[0], offsets[1] + 1):
                            end_token_idx = token_idx

                starting_token_idxs.append(start_token_idx)
                ending_token_idxs.append(end_token_idx)
                # print("start_token_idx", start_token_idx, "end_token_idx", end_token_idx)
            else:
                starting_token_idxs.append(0)
                ending_token_idxs.append(0)
        

        input["start_positions"] = starting_token_idxs
        input["end_positions"] = ending_token_idxs
        return  input

    return char2token_mapping

if __name__ == "__main__":
    datasets = load_dataset("stanfordnlp/coqa")

    # print(datasets)

    processor = ExtractiveQAPreProcesing()
    data = datasets["train"][:1]
    print("Raw Data:", data["answers"])
    result = processor(data)
    # pprint(processor(data))

# Train model 