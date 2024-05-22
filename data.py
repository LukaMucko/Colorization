import os
import glob
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from skimage import io, color
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def lab_to_rgb(l, ab):
    #One image at a time
    l, ab = l.cpu().numpy(), ab.detach().cpu().numpy()
    l = (l+1.) * 50
    ab = ab * 110 #Largest possible ab value is |-107.85730020669489|
    lab = np.concatenate([l, ab], axis=0).transpose(1, 2, 0)
    rgb = color.lab2rgb(lab)
    return rgb

def plot_images(dataloader):
    gray_imgs, color_imgs = next(iter(dataloader))
    num_imgs_to_plot = color_imgs.size(0) // 4
    fig, axs = plt.subplots(nrows=2, ncols=num_imgs_to_plot, figsize=(20, 6))

    for i in range(num_imgs_to_plot):
        img = lab_to_rgb(gray_imgs[i], color_imgs[i])
        
        ax = axs[0, i]
        ax.imshow(np.squeeze(gray_imgs[i], 0), cmap='gray')
        ax.axis('off')
        
        
        ax = axs[1, i]
        ax.imshow(img)
        ax.axis('off')

    plt.show()
    plt.close() 
    
def load_npy_data(color_path, gray_path, size=256, n=100, batch_size=32, test=True, p_train=0.8, num_workers=4, pin_memory=False):
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


def visualize(generator, dataloader, device="cuda", n=5):
    generator.to(device)
    generator.eval()
    gray_imgs, color_imgs = next(iter(dataloader))
    n = min(n, len(gray_imgs))
    gray_imgs = gray_imgs[:n]
    color_imgs = color_imgs[:n]

    gray_imgs = gray_imgs.to(device)
    with torch.no_grad():
        pred_ab = generator(gray_imgs).detach()
    generator.train()
    fig, axs = plt.subplots(nrows=3, ncols=n, figsize=(5*n, 10))

    if n == 1:
        axs = np.expand_dims(axs, axis=0)

    for i in range(n):
        ax = axs[0, i]
        ax.imshow(np.squeeze(gray_imgs[i].cpu().numpy(), 0), cmap='gray')
        ax.set_title('Grayscale Image')
        ax.axis('off')
        
        # Generated color image
        gen_img = lab_to_rgb(gray_imgs[i], pred_ab[i])
        ax = axs[1, i]
        ax.imshow(gen_img)
        ax.set_title('Generated Color Image')
        ax.axis('off')
        
        # Real color image
        real_img = lab_to_rgb(gray_imgs[i], color_imgs[i])
        ax = axs[2, i]
        ax.imshow(real_img)
        ax.set_title('Real Color Image')
        ax.axis('off')
    
    plt.show()
    plt.close()

class CocoDataset(Dataset):
    def __init__(self, paths, size=256):
        self.paths = paths
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             #transforms.Lambda(lambda x: 2*x - 1),
                                             transforms.Resize((size, size), antialias=True)])
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        img = color.rgb2lab(Image.open(self.paths[idx]).convert("RGB")).astype("float32")
        l = img[:,:,0] / 50. - 1.
        ab = img[:,:,1:] / 110.
        return self.transform(l), self.transform(ab)


def load_coco(n=1000, path="/lustre/home/lmucko/.fastai/data/coco_sample/train_sample/", size=256, batch_size=32, num_workers=4, pin_memory=False, load_all=False):
    paths = glob.glob(path + "/*.jpg") # Grabbing all the image file names
    np.random.seed(123)
    paths_subset = np.random.choice(paths, 10_000, replace=False) # choosing 1000 images randomly
    rand_idxs = np.random.permutation(10_000)
    train_idxs = rand_idxs[:8000] # choosing the first 8000 as training set
    val_idxs = rand_idxs[8000:] # choosing last 2000 as validation set
    train_paths = paths_subset[train_idxs]
    test_paths = paths_subset[val_idxs]
    print(test_paths)
    print(len(train_paths), len(test_paths))
    if not load_all:
        train_dataset = CocoDataset(train_paths, size)
        test_dataset = CocoDataset(test_paths, size)
        
        return DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=64, pin_memory=pin_memory), DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=64, pin_memory=pin_memory)   
