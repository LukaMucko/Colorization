import os
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from skimage import io, color
import matplotlib.pyplot as plt
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go

def lab_to_rgb(l, ab):
    #One image at a time
    l, ab = l.cpu().numpy(), ab.detach().cpu().numpy()
    l = (l+1.)/2 * 100
    ab = (ab+1.)/2 * 255 - 128
    lab = np.concatenate([l, ab], axis=0).transpose(1, 2, 0)
    rgb = color.lab2rgb(lab)
    return rgb

def plot_images(dataloader):
    gray_imgs, color_imgs = next(iter(dataloader))
    num_imgs_to_plot = color_imgs.size(0) // 4
    fig, axs = plt.subplots(nrows=num_imgs_to_plot, ncols=2, figsize=(10, 20))

    for i in range(num_imgs_to_plot):
        img = lab_to_rgb(gray_imgs[i], color_imgs[i])
        
        ax = axs[i, 0]
        ax.imshow(img)
        ax.set_title('Color Image')
        ax.axis('off')

        ax = axs[i, 1]
        ax.imshow(np.squeeze(gray_imgs[i], 0), cmap='gray')
        ax.set_title('Grayscale Image')
        ax.axis('off')

    plt.show()
    plt.close() 
    
def load_npy_data(color_path, gray_path, size=224, n=100, batch_size=32, test=True, p_train=0.8, num_workers=4, pin_memory=False):
    color_data = np.load(color_path)
    gray_data = np.load(gray_path)
    if "ab2.npy" in color_path:
        gray_data = gray_data[10000:]
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Lambda(lambda x: 2*x - 1), 
                                    transforms.Resize((size,size), antialias=True)])

    n = min(n, len(color_data), len(gray_data))
    color_data = color_data[:n] #ToTensor already normalizes to [0,1]
    gray_data = gray_data[:n]

    indices = np.random.choice(len(color_data), n, replace=False)
    
    color_data = color_data[indices]
    gray_data = gray_data[indices]
    
    if test:
        train_size = int(n * p_train)
        
        train_indices = np.random.choice(n, size=train_size, replace=False)
        test_indices = np.setdiff1d(np.arange(n), train_indices)

        color_data_train = color_data[train_indices]
        gray_data_train = gray_data[train_indices]
        
        color_data_test = color_data[test_indices]
        gray_data_test = gray_data[test_indices]

        train_dataset = [(transform(gray), transform(color)) for color, gray in zip(color_data_train, gray_data_train)]
        
        test_dataset = [(transform(gray), transform(color)) for color, gray in zip(color_data_test, gray_data_test)]
        
        return DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory), DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
        
    color_dataset = [(transform(gray), transform(color)) for color, gray in zip(color_data, gray_data)]

    return DataLoader(color_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)


def visualize(generator, dataloader, device="cuda", n=2):
    generator.to(device)
    generator.eval()
    gray_imgs, color_imgs = next(iter(dataloader))
    n = min(n, len(gray_imgs))
    gray_imgs = gray_imgs[:n]
    color_imgs = color_imgs[:n]

    gray_imgs = gray_imgs.to(device)
    pred_ab = generator(gray_imgs).detach()
    
    fig, axs = plt.subplots(nrows=n, ncols=3, figsize=(15, 5*n))

    if n == 1:
        axs = np.expand_dims(axs, axis=0)

    for i in range(n):
        ax = axs[i, 0]
        ax.imshow(np.squeeze(gray_imgs[i].cpu().numpy(), 0), cmap='gray')
        ax.set_title('Grayscale Image')
        ax.axis('off')
        
        # Generated color image
        gen_img = lab_to_rgb(gray_imgs[i], pred_ab[i])
        ax = axs[i, 1]
        ax.imshow(gen_img)
        ax.set_title('Generated Color Image')
        ax.axis('off')
        
        # Real color image
        real_img = lab_to_rgb(gray_imgs[i], color_imgs[i])
        ax = axs[i, 2]
        ax.imshow(real_img)
        ax.set_title('Real Color Image')
        ax.axis('off')
    
    plt.show()
    plt.close()

