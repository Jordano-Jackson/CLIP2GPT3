from transformers import AutoTokenizer
import transformers
import torch

class LLaMa2():
    def __init__(self):
        self.model = "meta-llama/Llama-2-7b-hf"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model)
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=self.model,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.freeze_model()

    def freeze_model(self):
        for param in self.pipeline.model.parameters():
            param.requires_grad = False
    
    def inference(self, embed, query):
        sequences = self.pipeline(
            embed + query,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            max_length=20,
        )
        for seq in sequences:
            print(f"Result: {seq['generated_text']}")


if __name__=='__main__':
    test = LLaMa2()
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