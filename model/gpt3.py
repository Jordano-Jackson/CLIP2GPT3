
from transformers import AutoTokenizer, GPT2LMHeadModel
import transformers

import torch.nn as nn
import torch
import numpy as np
import torch

class GPT3(nn.Module):
    def __init__(self):
        super(GPT3, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.model = GPT2LMHeadModel.from_pretrained("gpt2")
        #self.model = self.model.to(self.device)
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=self.model, #on cuda:0
            tokenizer=self.tokenizer,
            torch_dtype=torch.bfloat16,
            #device_map="auto",
            max_new_tokens=4,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            device = self.device
        )
        #self.freeze_model()

    def freeze_model(self):
        for param in self.pipeline.model.parameters():
            param.requires_grad = False
    
    def forward(self, embed, query):
        """
        :params:    embed(torch tensor(batch_size, 1024))
                    query(string list(batch_size))
        
        """
        self.tokenizer.pad_token = self.tokenizer.eos_token
        #tokenized_query = self.tokenizer(query, return_tensors='pt',padding=True, truncation=True)
        embed_str = [" ".join([str(round(element, 2)) for element in sublist]) for sublist in embed.tolist()]

        # Make a concatenated input created by concatenating the clip output and query
        # combined_input: list of string(batch_size, )
        combined_input = [f"Question: my question is : {a} \n\n and the image embedding that correspond to the question is [{b}]. \n\n Please answer Yes or No.\n\n Answer: I think the answer is " for a, b in zip(query, embed_str)]

        ## text input must be a string 
        sequences = self.pipeline(
            text_inputs=combined_input,
            top_k=10,
            #num_return_sequences=1,
        )
        
        # Remove duplicated question in the answer
        for i in range(len(sequences)):
            sequences[i][0]['generated_text'] = sequences[i][0]['generated_text'][len(combined_input[i]):]
            #print(combined_input,sequences[i][0]['generated_text'], )
        #for seq in sequences:
            #print(f"Result: {seq[0]['generated_text']}")

        # The sequence is a list of [batch_size, 1]
        # that is a dictionary contains the generated_text as a key value
        return [sequence[0]['generated_text'] for sequence in sequences]
        

    def inference_text(self, text):
        # this method is made for debuggign purpose

        sequences = self.pipeline(
            text,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            
        )
        for seq in sequences:
            print(f"Result: {seq['generated_text']}")


if __name__=='__main__':
    print('initializing...')
    test = GPT3()
    print('initializing done.')
    while(True):
        text = str(input('input the text: '))
        test.inference_text(text)
    test.inference('','hi my name is ')

"""
model = "meta-llama/Llama-2-7b-hf"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)

sequences = pipeline(
    'i\'m doin test',
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    max_length=20,
)
for seq in sequences:
    print(f"Result: {seq['generated_text']}")
"""