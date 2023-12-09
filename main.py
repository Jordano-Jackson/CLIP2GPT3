import itertools
import time
import argparse

import torch.optim as optim
import torch.nn as nn
import torch
from torch.utils.data import DataLoader

from model.model import MixModel
from dataset.vqa_dataset import VQADataset

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mix_model = MixModel()
    mix_model = mix_model.to(device)
    if args.load_state_dict:
        print("Loading params..")
        mix_model.load_state_dict(torch.load(args.load_state_dict))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(mix_model.parameters(), lr=2e-5)

    num_epochs = 5

    start_time = time.time()
    print("Loading Dataset..")
    train_dataset = VQADataset('yes/no')

    train_dataloader = DataLoader(dataset= train_dataset, batch_size=2, shuffle=True)
    print("Loading Dataset Done.")

    for epoch in range(num_epochs):
        mix_model.train()
        total_loss = 0.0
        step_loss = 0.0

        print("Training Start..")
        for step, (query, image, label) in enumerate(train_dataloader, 1):  
            """
            query: <class 'tuple'> 
            image: <class 'torch.Tensor'> 
            label: <class 'tuple'>

            query ex: ('Is there anywhere to change a baby diaper in this room?', 'Is the toilet neat?')
            image ex: tensor([[[[0.2784, 0.2784, 0.2745,  ..., 0.9020, 0.8980, 0.8980],
            label ex: ('yes', 'no')
            """
            #query=query.to(device)
            image = image.to(device)            
            label = torch.tensor([1 if 'yes' in item else 0 for item in label])
            label=label.to(device)

            optimizer.zero_grad()

            output = mix_model(query,image)

            loss = criterion(output, label)

            loss.backward()

            if step % 100 == 0:
                elapsed_time = time.time() - start_time
                print(f"Epoch {epoch+1}/{num_epochs}, Step {step}/{len(train_dataloader)}, Batch Loss: {step_loss/100}, Elapsed Time: {elapsed_time}s")
                step_loss =0.0

            if step % 1000 == 0:
                print("Saving the params...")
                torch.save(mix_model.state_dict(), "params/params.pth")
                print("Saving the params done.")

            step_loss += loss.item()
            total_loss += loss.item()
            optimizer.step()

        average_loss = total_loss / len(train_dataloader)
        elapsed_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {average_loss}, Elapsed Time: {elapsed_time} s")
        torch.save(mix_model.state_dict(), "params.pth")

def test(arg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mix_model = MixModel()
    mix_model = mix_model.to(device)
    print("Loading params..")
    mix_model.load_state_dict(torch.load(args.load_state_dict))

    print("Loading Dataset..")
    test_dataset = VQADataset('yes/no', type='test')

    test_dataloader = DataLoader(dataset= test_dataset, batch_size=2, shuffle=True)
    print("Loading Dataset Done.")

    mix_model.eval() 

    correct_predictions = 0
    total_samples = 0
    
    limited_test_dataloader = itertools.islice(test_dataloader, 2)

    with torch.no_grad():
        for step, (query, image, label) in enumerate(limited_test_dataloader, 1): 
            image = image.to(device)
            label = torch.tensor([1 if 'yes' in item else 0 for item in label])
            label = label.to(device)

            output = mix_model(query, image)
            _, predicted = torch.max(output, 1)

            total_samples += label.size(0)
            correct_predictions += (predicted == label).sum().item()
            
            if step % 100 == 0 :
                print(f"Current step: {step/1000}")

    accuracy = correct_predictions / total_samples
    print(f"Test Accuracy: {accuracy}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training and Testing arguments")
    parser.add_argument("--load_state_dict", type=str, help="Path to the saved state_dict file")
    parser.add_argument("--mode", choices=["train", "test"], help="Select mode: train or test")

    args = parser.parse_args()

    if args.mode == "train":
        train(args)
    elif args.mode == "test":
        test(args)
    else:
        print("Invalid mode. Please choose 'train' or 'test'.")