import requests

from PIL import Image
import numpy as np
import torch
import torch.nn as nn 

from transformers import CLIPProcessor, CLIPModel, AutoTokenizer

class CLIP(nn.Module):
    def __init__(self): 
        super(CLIP, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

        # Freeze model
        for param in self.model.parameters():
            param.requires_grad = False

        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    def forward(self, image, query):
        """
        Inference and return the result
        
        params: image(PIL image)
                text(str)

        return: output(torch tensor (n*1024))
        """
        input = self.processor(text=query, images=image, return_tensors="pt", padding=True)
        input = input.to(self.device)
        output = self.model(**input)
        text_embed = output[2]
        image_embed = output[3]
        return torch.concat([text_embed, image_embed], dim=1)
    
    def get_text_embed(self, text):
        input = self.tokenizer(text=text, return_tensors="pt", padding=True).to(self.device)
        output = self.model.get_text_features(**input)
        return output
        
if __name__ == "__main__":
    clip = CLIP()
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    output =clip.forward(image, ["a photo of a cat"])
    print(output.shape)
    #logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
    #probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
