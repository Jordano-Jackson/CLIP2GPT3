import requests

from PIL import Image
import numpy as np
import torch

from transformers import CLIPProcessor, CLIPModel

class CLIP():
    def __init__(self): 
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        for param in self.model.parameters():
            param.requires_grad = False
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def inference(self, image, query):
        """
        Inference and return the result
        
        params: image(PIL image)
                text(str)

        return: output(torch tensor (n*1024))
        """
        input = self.processor(text=query, images=image, return_tensors="pt", padding=True)
        output = self.model(**input)
        text_embed = output[2]
        image_embed = output[3]
        print(type(text_embed))
        return torch.concat([text_embed, image_embed], dim=1)



if __name__ == "__main__":
    clip = CLIP()
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    output =clip.inference(image, ["a photo of a cat"])
    print(output.shape)
    #logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
    #probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
