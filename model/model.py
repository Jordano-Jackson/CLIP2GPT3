import torch
import torch.nn as nn

from mtransformer import MixTransformer
from llama2 import LLaMa2
from clip import CLIP

class MixModel(nn.Module):
    def __init__(self):
        super(MixModel, self).__init__()
        self.mtransformer = MixTransformer(
            input_size = 1024,
            head_size = 64,
            hidden_size = 2048,
            num_layers = 6
        )
        self.llama2_model = LLaMa2()
        self.clip_model = CLIP()

    def forward(self, query, image):

        clip_output = self.clip_model.inference(image,query)
        mixtrans_output = self.mtransformer.forward(clip_output)
        output = self.llama2_model.inference(mixtrans_output, query)
        
        return output
