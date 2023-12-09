import torch
import torch.nn as nn
import numpy as np

from model.mtransformer import MixTransformer
from model.llama2 import LLaMa2
from model.gpt3 import GPT3
from model.clip import CLIP

class MixModel(nn.Module):
    def __init__(self):
        super(MixModel, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.mtransformer = MixTransformer(
            input_size = 1024,
            head_size = 16,
            hidden_size = 1024,
            num_layers = 6
        ).to(self.device)
        self.ffnn = nn.Sequential(
            nn.Linear(1024, 256),
            nn.Linear(256,64),
        ).to(self.device)
        self.binary = nn.Linear(512,2).to(self.device)

        self.clip_model = CLIP().to(self.device)
        self.llm = GPT3().to(self.device)

    def forward(self, query, image, ansType="yes/no"):
    
        clip_output = self.clip_model.forward(image,query)
        clip_output = clip_output.to("cuda" if torch.cuda.is_available() else "cpu")

        mixtrans_output = self.mtransformer.forward(clip_output)
        mixtrans_output = self.ffnn(mixtrans_output)
        output = self.llm.forward(mixtrans_output, query)
        
        if ansType=='yes/no':
            output = self.binary_answer(output)
        return output
    
    def binary_answer(self, sequences, inference=False):
        """
        :params:    sequences(list of string[batch_size,])

        :output:    list of int[batch_size] -> 1: True, 0: False
        """
        
        # The maximum input size of CLIP text embedder is (77)
        text_embed = self.clip_model.get_text_embed(sequences).to(self.device)
        text_embed = self.binary(text_embed)
        ans=text_embed
        if inference:
            _, ans = torch.max(text_embed, dim=1)
        return ans 

            
