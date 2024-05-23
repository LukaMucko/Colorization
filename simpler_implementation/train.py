import torch
import torch.nn as nn
import torch.optim as optim
import data
from d2l import torch as d2l
import matplotlib.pyplot as plt
from models import init_weights

class LitModel():
    def __init__(self, generator, discriminator, lambda_recon=100):
        self.generator = generator
        self.discriminator = discriminator
        init_weights(self.generator)
        init_weights(self.discriminator)
        self.gen_criterion = nn.L1Loss()
        self.disc_criterion = nn.BCEWithLogitsLoss()
        self.gen_optimizer = optim.Adam(self.generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
        self.disc_optimizer = optim.Adam(self.discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lambda_recon = lambda_recon
        self.generator = self.generator.to(self.device)
        self.discriminator = self.discriminator.to(self.device)
        self.accumulator = d2l.Accumulator(2)
        
    def set_requires_grad(self, model, requires_grad=True):
        for param in model.parameters():
            param.requires_grad = requires_grad    
    
    def discriminator_step(self, trainL, trainAB, fakeAB):
        fake_pred = self.discriminator(trainL, fakeAB)
        real_pred = self.discriminator(trainL, trainAB)
        fake_loss = self.disc_criterion(fake_pred, torch.zeros_like(fake_pred).to(self.device))
        real_loss = self.disc_criterion(real_pred, torch.ones_like(real_pred).to(self.device))
        disc_loss = 0.5 * (fake_loss + real_loss)
        return disc_loss
    
    def generator_step(self, trainL, trainAB, fakeAB):
        fake_output = self.discriminator(trainL, fakeAB)
        adv_loss = self.disc_criterion(fake_output, torch.ones_like(fake_output).to(self.device))
        recon_loss = self.gen_criterion(fakeAB, trainAB) * self.lambda_recon
        return recon_loss + adv_loss
    
    def training_step(self, batch):
        trainL, trainAB = batch
        trainL, trainAB = trainL.to(self.device), trainAB.to(self.device)
        
        #Discriminator Step
        self.disc_optimizer.zero_grad()
        self.set_requires_grad(self.discriminator, True)
        fakeAB = self.generator(trainL)
        disc_loss = self.discriminator_step(trainL, trainAB, fakeAB.detach())    
        disc_loss.backward()
        self.disc_optimizer.step()
        
        #Generator Step
        self.gen_optimizer.zero_grad()
        self.set_requires_grad(self.discriminator, False)
        gen_loss = self.generator_step(trainL, trainAB, fakeAB)
        gen_loss.backward()
        self.gen_optimizer.step()
        
        self.accumulator.add(gen_loss.item(), disc_loss.item())
        
        return gen_loss.item(), disc_loss.item()
        
    def save_model(self, path, discriminator_path = None):
        torch.save(self.generator.state_dict(), path)
        if discriminator_path:
            torch.save(self.discriminator.state_dict(), discriminator_path)
        
    def visualize(self, batch):
        data.visualize(self.generator, batch)
        
    def plot_losses(self):
        plt.plot(self.accumulator[0], label='Generator Loss')
        plt.plot(self.accumulator[1], label='Discriminator Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
        
    def load_model(self, path, discriminator_path = None):
        print(self.generator.load_state_dict(torch.load(path)))
        if discriminator_path:
            print(self.discriminator.load_state_dict(torch.load(discriminator_path)))