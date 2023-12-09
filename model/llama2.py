from transformers import AutoTokenizer
import transformers
import torch
import numpy as np

class LLaMa2():
    def __init__(self):
        self.model = "meta-llama/Llama-2-7b-hf"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model)
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=self.model,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.freeze_model()

    def freeze_model(self):
        for param in self.pipeline.model.parameters():
            param.requires_grad = False
    
    def inference(self, embed, query):
        """
        :params:    embed(torch tensor(batch_size, 1024))
                    query(string list(batch_size))
        
        """
        with torch.autocast("cuda"):
            self.tokenizer.pad_token = self.tokenizer.eos_token
            #tokenized_query = self.tokenizer(query, return_tensors='pt',padding=True, truncation=True)
            embed_str = [" ".join([str(element) for element in sublist]) for sublist in embed.tolist()]

            # Make a concatenated input created by concatenating the clip output and query
            # combined_input: list of string(batch_size, )
            combined_input = [f"the query is : {a} \n\n and the image embedding is [{b}]" for a, b in zip(query, embed_str)]
            
            ## text input must be a string 
            sequences = self.pipeline(
                text_inputs=combined_input,
                top_k=10,
                num_return_sequences=1,
                eos_token_id=self.tokenizer.eos_token_id,
                max_length=20,
            )
            for seq in sequences:
                print(f"Result: {seq['generated_text']}")

    def inference_text(self, text):
        # this method is made for debuggign purpose
        tokenizer = AutoTokenizer.from_pretrained(self.model)
    

        sequences = self.pipeline(
            text,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            max_length=20,
        )
        for seq in sequences:
            print(f"Result: {seq['generated_text']}")


if __name__=='__main__':
    print('initializing...')
    test = LLaMa2()
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