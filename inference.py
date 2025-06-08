import torch
from transformers import RobertaTokenizerFast
from utils import RobertaConfig
from model import RobertaForQuestionAnswering
from safetensors.torch import load_file
from datasets import load_dataset
from pprintpp import pprint



class InferenceModel:
    """
        Quick inference function that works with the models we have trained!
    """

    def __init__(self, path_to_weights, huggingface_model=True):
        ### Init Config with either Huggingface Backbone or our own ###
        self.config = RobertaConfig(pretrained_backbone="pretrained_huggingface" if huggingface_model else "random")

        ### Load Tokenizer ###
        self.tokenizer = RobertaTokenizerFast.from_pretrained(self.config.hf_model_name)

        ### Load Model ###
        self.model = RobertaForQuestionAnswering(self.config)

        weights = load_file(path_to_weights)
        self.model.load_state_dict(weights)

        self.model.eval()

    def inference_model(self,
                        question,
                        context):
        ### Tokenize Text
        inputs = self.tokenizer(text=question,
                                text_pair=context,
                                max_length=self.config.context_length,
                                truncation="only_second",
                                return_tensors="pt")
        pass
        ### Pass through Model ####
        with torch.no_grad():
            start_token_logits, end_token_logits = self.model(**inputs)

        ### Grab Start and End Token Idx ###
        start_token_idx = start_token_logits.squeeze().argmax().item()
        end_token_idx = end_token_logits.squeeze().argmax().item()


        ### Slice Tokens and then Decode with Tokenizer (+1 because slice is not right inclusive) ###
        tokens = inputs["input_ids"].squeeze()[start_token_idx:end_token_idx + 1]
        answer = self.tokenizer.decode(tokens, skip_special_tokens=True).strip()

        prediction = {"start_token_idx": start_token_idx,
                      "end_token_idx": end_token_idx,
                      "answer": answer}

        return prediction


if __name__ == "__main__":

    dataset = load_dataset("stanfordnlp/coqa")
    
    data = dataset["validation"][2]
    # data = dataset["train"][0]
    # print("answer:", data["answers"])
    ### Sample Text ###
    context = data["story"]
    print("context:", context)
    question = data["questions"][4]

    tokenizer = RobertaTokenizerFast.from_pretrained("deepset/roberta-base-squad2")

    encoded = tokenizer(
        question,
        context,
        max_length=512,
        truncation="only_second",
        padding="max_length",
        return_offsets_mapping=True,
        return_tensors="pt"
    )
    offset_mapping = encoded["offset_mapping"][0].tolist()  # convert to list of tuples
    input_ids = encoded["input_ids"][0]


    ### Inference Model ###
    path_to_weights = "model/RoBERTa/save_model/model.safetensors" 
    inferencer = InferenceModel(path_to_weights=path_to_weights, huggingface_model=True)
    prediction = inferencer.inference_model(question, context)
    print("\n----------------------------------")
    print("results:", prediction)

    start_token_idx = prediction["start_token_idx"]
    end_token_idx = prediction["end_token_idx"]

    start_char = offset_mapping[start_token_idx][0]
    end_char = offset_mapping[end_token_idx][1]
    
    print("Question:", question)
    print("Recovered answer:", context[start_char:end_char])

    # test model