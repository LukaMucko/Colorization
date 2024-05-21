import torch
from fastai.vision.learner import create_body
from fastai.vision.models.unet import DynamicUnet
from rich.progress import Progress
import torchvision
from data import load_npy_data
import argparse
from models import UNet
import os

def build_unet_from_model(model, n_output=2, size=256):
    body = create_body(model(), 1, True, -2)
    generator = DynamicUnet(body, n_output, (size, size))
    generator = torch.nn.Sequential(generator, torch.nn.Tanh())
    return generator

def init_weights(model):
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.ConvTranspose2d):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)

def train(generator, trainLoader, epochs=20):
    generator.to("cuda")
    optim = torch.optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    loss_fn = torch.nn.L1Loss()
    
    with Progress() as progress:
        task = progress.add_task("[red]Training...", total=epochs)
        inner_progress = progress.add_task("[yellow]Training...", total=len(trainLoader))
        for epoch in range(epochs):
            loss = 0
            n = 0
            for x, y in trainLoader:
                x, y = x.cuda(), y.cuda()
                optim.zero_grad()
                y_pred = generator(x)
                l = loss_fn(y_pred, y)
                l.backward()
                optim.step()
                
                loss += l.item()
                n += x.size(0)
                progress.update(inner_progress, advance=1, description=f"Loss: {loss / n}")
            progress.reset(inner_progress, total=len(trainLoader))
            progress.update(task, advance=1)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, required=False, help="The model to use (e.g., 'resnet18')")
    parser.add_argument("--save_path", type=str, required=True, help="The path to save the model (e.g., 'Models/pretrained/resnet18.pth')")
    parser.add_argument("--epochs", type=int, default=20, help="The number of epochs to train for (default: 20)")
    parser.add_argument("--n_images", type=int, default=2000, help="The number of images to use for training (default: 2000)")
    parser.add_argument("--UNet", type=bool, default=False, help="Whether to use a UNet model from models.py (default: False)")
    args = parser.parse_args()
    if args.model:
        model = getattr(torchvision.models, args.model)
    dl = load_npy_data("ab/ab/ab1.npy", "l/gray_scale.npy", size=256, n=args.n_images, batch_size=32, test=False)
    
    if not args.UNet:
        generator = build_unet_from_model(model, n_output=2, size=256)
        print(f"Generator is UNet with {args.model} body")
        name = f"{args.save_path}/pretrained/unet_{args.model}_{args.epochs}_{args.n_images}.pth"
    else:
        generator = UNet(in_channels=1, out_channels=2, image_size=256)
        print(f"Generator is UNet from models.py")
        name = f"{args.save_path}/pretrained/generator_{args.epochs}_{args.n_images}.pth"
    print("Saving at path", name)
    init_weights(generator)
    
    train(generator, dl, epochs=args.epochs)
    
    if not os.path.exists(args.save_path+"/pretrained"):
        os.makedirs(args.save_path+"/pretrained")
        
    torch.save(generator.state_dict(), name)
    print("Saved model state dictionary at", name)
    
if __name__ == "__main__":
    main()    
    