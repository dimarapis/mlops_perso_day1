import argparse
import sys

import torch
import click

from data import mnist
from model import MyAwesomeModel
import matplotlib.pyplot as plt
import cv2

@click.group()
def cli():
    pass


@click.command()
@click.option("--lr", default=1e-3, help='learning rate to use for training')

def train(lr):
    print("Training day and night")
    print(lr)


    model = MyAwesomeModel()
    train_set, _ = mnist()
    print(train_set)
    criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    
    epochs = 30
    running_losses = []
    for e in range(epochs):
        model.train()
        running_loss = 0
        for images, labels in train_set:
            optimizer.zero_grad()        
            log_ps = model(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()           
            running_loss += loss.item()
        running_losses.append(running_loss/len(train_set))   
        print(f'Epoch {e} --- train loss: {running_loss/len(train_set)}') 
    
    torch.save(model.state_dict(), 'checkpoint.pth')
    
    
        #else:
        #    with torch.no_grad():
        #        model.eval()
        #        images, labels = test_set
        #        log_ps = model(images)
        #        _, top_class = log_ps.topk(1, dim=1)
        #        equals = top_class == labels.view(*top_class.shape)
        #        accuracy = torch.mean(equals.type(torch.FloatTensor))
        #        loss = criterion(log_ps, labels)/images.shape[0]
        #        test_losses.append(loss)
        #        print(f'{e}: train loss: {running_loss/steps}, test loss: {loss}, test accuracy: {accuracy}')
        #train_losses.append(running_loss)
        
    plt.figure(figsize=(8,4))
    plt.plot(running_losses, label='training loss')
    #plt.plot(test_losses, label='test')
    plt.legend()
    plt.show()



@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    
    print("Evaluating until hitting the ceiling")
    
    print(model_checkpoint)
    _, test_set = mnist()
    #print(len(test_set[0]))
    #print(len(test_set[1]))
    
    #.shape)
    model = MyAwesomeModel()

    with torch.no_grad():
        model.eval()
    model.load_state_dict(torch.load(model_checkpoint))
    criterion = torch.nn.NLLLoss()
    
    test_losses = []
    test_loss = 0
    images,labels = test_set
    #for images, labels in test_set:
    log_ps = model(images)
    loss = criterion(log_ps, labels)
    test_loss += loss.item()
    test_losses.append(loss.item())
            
    ps = torch.exp(log_ps)
    top_p, top_class = ps.topk(1, dim=1)
    equals = top_class == labels.view(*top_class.shape)
    accuracy = torch.mean(equals.type(torch.FloatTensor))
    #print(test_loss)  
    print(f'Accuracy: {accuracy.item()*100}%')
    
    
    
    
cli.add_command(train)
cli.add_command(evaluate)

if __name__ == "__main__":
    cli()


    
    
    
    