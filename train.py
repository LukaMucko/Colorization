import torch
from rich.progress import Progress #ima se, može se
from data import visualize, load_npy_data
import numpy as np
from models import UNet, Discriminator
import argparse
import os
import json

def gradient_penalty(discriminator, real_ab, fake_ab, L, device):
    alpha = torch.rand(real_ab.size(0), 1, 1, 1, device=device)
    x_hat = (alpha * real_ab + (1 - alpha) * fake_ab).float().requires_grad_(True)
    out = discriminator(L, x_hat)
    gradients = torch.autograd.grad(outputs=out, inputs=x_hat,
                                    grad_outputs=torch.ones(out.size(), device=device),
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradients_norm = (gradients+1e-16).norm(2, dim=1)
    return ((gradients_norm - 1) ** 2).mean()

def set_requires_grad(model, requires_grad=True):
    for param in model.parameters():
        param.requires_grad = requires_grad

def train_wgan(generator, discriminator, train_loader, optimizer_gen, optimizer_disc, lambda_recon=100, lambda_gp=10, device='cuda', epochs=10, save_model=False, save_path='Models', visualize_every=-1, plot_images=False, n=400):
    generator.to(device)
    discriminator.to(device)
    generator.train()
    discriminator.train()
    l1_loss = torch.nn.L1Loss()

    gen_losses, disc_losses = [], []

    if visualize_every < 1:
        plot_images = False

    with Progress() as progress:
        outer_task = progress.add_task("[cyan]Epochs", total=epochs)
        inner_progress = progress.add_task("[yellow]Training...", total=len(train_loader))
        
        for epoch in range(1, epochs + 1):
            running_gen_loss = 0.0
            running_disc_loss = 0.0
            
            for i, (trainL, trainAB) in enumerate(train_loader):
                trainL, trainAB = trainL.to(device), trainAB.to(device)
                
                # Train Discriminator 
                set_requires_grad(discriminator, True)
                optimizer_disc.zero_grad()
                
                real_output = discriminator(trainL, trainAB)
                real_loss = -torch.mean(real_output)
                
                fakeAB = generator(trainL)
                fake_output = discriminator(trainL, fakeAB.detach())
                fake_loss = torch.mean(fake_output)
                
                # Gradient penalty
                gp = gradient_penalty(discriminator, trainAB, fakeAB.detach(), trainL, device)
                
                disc_loss = real_loss + fake_loss + lambda_gp * gp
                disc_loss.backward()
                optimizer_disc.step()
                
                disc_losses.append(disc_loss.item())
                running_disc_loss += disc_loss.item()
                
                # Train Generator  
                set_requires_grad(discriminator, False)  
                optimizer_gen.zero_grad()    
                
                fake_output = discriminator(trainL, fakeAB)
                adv_loss = -torch.mean(fake_output)
                recon_loss = l1_loss(fakeAB, trainAB)
                
                gen_loss = lambda_recon * recon_loss + adv_loss
                gen_loss.backward()
                optimizer_gen.step()
                
                gen_losses.append(gen_loss.item())
                running_gen_loss += gen_loss.item()
            
                progress.update(inner_progress, advance=1, description=f"[cyan]Training... [Gen Loss: {running_gen_loss / (i + 1):.4f}] [Disc Loss: {running_disc_loss / (i + 1):.4f}]")
                    
            if plot_images and (epoch % visualize_every == 0 or epoch == 1):
                visualize(generator, train_loader, device=device, n=1)

            progress.reset(inner_progress, total=len(train_loader))
            progress.update(outer_task, advance=1)
            
    if save_model:
        save = f"{save_path}/WGAN_{epochs}_{lambda_recon}_{n}"
        if not os.path.exists(save):
            os.makedirs(save)
        torch.save(generator.state_dict(), f"{save}/generator.pth")
        torch.save(discriminator.state_dict(), f"{save}/discriminator.pth")
        
        np.save(f"{save}/generator_losses.npy", np.array(gen_losses))
        np.save(f"{save}/discriminator_losses.npy", np.array(disc_losses))
        
        print("Done training:", "Models saved at", save)


def train_gan(generator, discriminator, train_loader, optimizer_gen, optimizer_disc, lambda_recon=100, device='cuda', epochs=10, save_model=False, save_path='Models', visualize_every=-1, plot_images=False, n=400):
    #Discriminator is a patch discriminator
    generator.to(device)
    discriminator.to(device)
    generator.train()
    discriminator.train()
    l1_loss = torch.nn.L1Loss()
    bce_loss = torch.nn.BCEWithLogitsLoss()
    if visualize_every < 1:
        plot_images = False

    gen_losses, disc_losses = [], []
    
    with Progress() as progress:
        outer_task = progress.add_task("[cyan]Epochs", total=epochs)
        inner_progress = progress.add_task("[yellow]Training...", total=len(train_loader))
        for epoch in range(1,epochs+1):
            running_gen_loss = 0.0
            running_disc_loss = 0.0
            
            for i, (trainL, trainAB) in enumerate(train_loader):
                trainL, trainAB = trainL.to(device), trainAB.to(device)
                
                # Train Discriminator 
                #real:
                set_requires_grad(discriminator, True)
                optimizer_disc.zero_grad()
                
                real_output = discriminator(trainL, trainAB)
                real_labels = torch.ones_like(real_output).to(device)
                fake_labels = torch.zeros_like(real_labels).to(device)
                real_loss = bce_loss(real_output, real_labels)
                
                #fake:
                fakeAB = generator(trainL)
                fake_output = discriminator(trainL, fakeAB.detach())
                fake_loss = bce_loss(fake_output, fake_labels)
                
                disc_loss = (real_loss + fake_loss) / 2
                disc_loss.backward()
                optimizer_disc.step()
                
                disc_losses.append(disc_loss.item())
                running_disc_loss+=disc_loss.item()
                
                # Train Generator  
                set_requires_grad(discriminator, False)  
                optimizer_gen.zero_grad()    
                
                fake_output = discriminator(trainL, fakeAB)
                adv_loss = bce_loss(fake_output, real_labels)
                recon_loss = l1_loss(fakeAB, trainAB)
                
                gen_loss = lambda_recon * recon_loss + adv_loss
                gen_loss.backward()
                optimizer_gen.step()
                
                
                gen_losses.append(gen_loss.item())
                running_gen_loss+=gen_loss.item()
            
            
                progress.update(inner_progress, advance=1, description=f"[cyan]Training... [Gen Loss: {running_gen_loss / (i + 1):.4f}] [Disc Loss: {running_disc_loss / (i + 1):.4f}]")
                    
            if plot_images and (epoch % visualize_every == 0 or epoch==1):
                        visualize(generator, train_loader, device=device, n=1)

            progress.reset(inner_progress, total=len(train_loader))
            progress.update(outer_task, advance=1)
            
    if save_model:
        save = f"{save_path}/GAN_{epochs}_{lambda_recon}_{args.lr_gen}_{args.lr_disc}_{n}"
        if not os.path.exists(save):
            os.makedirs(save)
        torch.save(generator.state_dict(), f"{save}/generator.pth")
        torch.save(discriminator.state_dict(), f"{save}/discriminator.pth")
        
    np.save(f"{save}/generator_losses.npy", np.array(gen_losses))
    np.save(f"{save}/discriminator_losses.npy", np.array(disc_losses))
    
    print("Done training:", "Models saved at", save)

def train_gan_argparser():
    parser = argparse.ArgumentParser(description="Train GAN function arguments")

    parser.add_argument("--type", type=str, default="gan", help="Type of GAN to train (gan or wgan)")
    parser.add_argument("--lr_gen", type=float, default=0.0002, help="Learning rate for the generator optimizer")
    parser.add_argument("--lr_disc", type=float, default=0.0002, help="Learning rate for the discriminator optimizer")
    parser.add_argument("--lambda_recon", type=float, default=100, help="Weight for the reconstruction loss")
    parser.add_argument("--device", type=str, default="cuda", help="Device to train the models on (cuda or cpu)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs for training")
    parser.add_argument("--verbose", type=bool, default=True, help="Whether to print training progress")
    parser.add_argument("--save_model", type=bool, default=True, help="Whether to save trained models")
    parser.add_argument("--save_path", type=str, default="Models", help="Path to save trained models")
    parser.add_argument("--n", type=int, default=1000, help="Number of images to visualize")
    parser.add_argument("--load_generator", type=str, default=None, help="Path to load generator model")
    parser.add_argument("--lambda_gp", type=float, default=10, help="Weight for the gradient penalty")
    parser.add_argument("--ab_path", type=str, default="ab/ab/ab1.npy", help="Path to AB images")
    parser.add_argument("--l_path", type=str, default="l/gray_scale.npy", help="Path to L images")

    return parser

def main(type):
    if type=="gan":
        train_gan(generator, discriminator, trainLoader, optimizer_gen, optimizer_disc,
              lambda_recon=args.lambda_recon, device=args.device, epochs=args.epochs,
              save_model=args.save_model, save_path=args.save_path,
              n=args.n, plot_images=False)
    else:
        train_wgan(generator, discriminator, trainLoader, optimizer_gen, optimizer_disc,
              lambda_recon=args.lambda_recon, lambda_gp=args.lambda_gp, device=args.device, epochs=args.epochs,
              save_model=args.save_model, save_path=args.save_path,
              plot_images=False)
        
if __name__ == "__main__":
    parser = train_gan_argparser()
    args = parser.parse_args()
    
    if args.type=="gan":
        discriminator=Discriminator(3)
    else:
        discriminator=Discriminator(3, patch=False)
    
    generator = UNet(1, 2)

    if args.load_generator:
        print(generator.load_state_dict(torch.load(args.load_generator)))
        
    
    optimizer_gen = torch.optim.Adam(generator.parameters(), lr=args.lr_gen, betas = (0.5, 0.999))
    optimizer_disc = torch.optim.Adam(discriminator.parameters(), lr=args.lr_disc, betas = (0.5, 0.999))
    trainLoader = load_npy_data(args.ab_path, args.l_path, size=256, n=args.n, batch_size=32, test=False, num_workers=4, pin_memory=True)
    
    print("Training...")
    print(json.dumps(args.__dict__, indent=4))
    main(args.type)
    
