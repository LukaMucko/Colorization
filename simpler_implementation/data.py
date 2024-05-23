import glob
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from skimage import io, color
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

class CocoDataset(Dataset):
    def __init__(self, paths):
        self.paths = paths
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256), antialias=True)
        ])

    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        img = color.rgb2lab(img).astype("float32")
        img = self.transform(img)
        l = img[[0], ...] / 50. - 1.
        ab = img[[1, 2], ...] / 110.
        return l, ab

def get_loaders(path="/lustre/home/lmucko/.fastai/data/coco_sample/train_sample", batch_size=16, num_workers=4, n=8000):
    np.random.seed(42069)
    paths = glob.glob(path+"/*.jpg")
    n = min(n, 10000)
    paths_subset = np.random.choice(paths, 10000, replace=False)
    rand_idxs = np.random.permutation(10000)
    train_idxs = rand_idxs[:n] 
    val_idxs = rand_idxs[n:]
    train_paths = paths_subset[train_idxs]
    val_paths = paths_subset[val_idxs]
    
    train_dataset = CocoDataset(train_paths)
    test_dataset = CocoDataset(val_paths)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    return train_loader, test_loader    
    
def lab2rgb(l, ab):
    l, ab = l.cpu().numpy(), ab.detach().cpu().numpy()
    l = (l + 1.) * 50.
    ab = ab * 110.
    lab = np.concatenate([l, ab], axis=0).transpose(1, 2, 0)
    rgb = color.lab2rgb(lab)
    return rgb

def show_batch(dataloader):
    gray_imgs, color_imgs = next(iter(dataloader))
    num_imgs_to_plot = color_imgs.size(0) // 4
    fig, axs = plt.subplots(nrows=2, ncols=num_imgs_to_plot, figsize=(20, 6))

    for i in range(num_imgs_to_plot):
        img = lab2rgb(gray_imgs[i], color_imgs[i])
        
        ax = axs[0, i]
        ax.imshow(np.squeeze(gray_imgs[i], 0), cmap='gray')
        ax.axis('off')
        
        
        ax = axs[1, i]
        ax.imshow(img)
        ax.axis('off')

    plt.show()
    plt.close() 

def visualize(generator, data, device="cuda", n=5, transpose=False):
    generator.to(device)
    generator.eval()
    if isinstance(data, DataLoader):
        gray_imgs, color_imgs = next(iter(data))
    else:
        gray_imgs, color_imgs = data
    n = min(n, len(gray_imgs))
    gray_imgs = gray_imgs[:n]
    color_imgs = color_imgs[:n]

    gray_imgs = gray_imgs.to(device)
    with torch.no_grad():
        pred_ab = generator(gray_imgs).detach()
    generator.train()
    
    if transpose:
        fig, axs = plt.subplots(nrows=n, ncols=3, figsize=(15, 5*n))
    else:
        fig, axs = plt.subplots(nrows=3, ncols=n, figsize=(5*n, 10))

    if n == 1:
        axs = np.expand_dims(axs, axis=0)

    for i in range(n):
        if transpose:
                        # Grayscale Image
            ax = axs[i, 0]
            ax.imshow(np.squeeze(gray_imgs[i].cpu().numpy(), 0), cmap='gray')
            ax.set_title('Grayscale Image')
            ax.axis('off')

            # Generated color image
            gen_img = lab2rgb(gray_imgs[i].cpu(), pred_ab[i].cpu())
            ax = axs[i, 1]
            ax.imshow(gen_img)
            ax.set_title('Generated Color Image')
            ax.axis('off')

            # Real color image
            real_img = lab2rgb(gray_imgs[i].cpu(), color_imgs[i].cpu())
            ax = axs[i, 2]
            ax.imshow(real_img)
            ax.set_title('Real Color Image')
            ax.axis('off')
        else:
            ax = axs[0, i]
            ax.imshow(np.squeeze(gray_imgs[i].cpu().numpy(), 0), cmap='gray')
            ax.set_title('Grayscale Image')
            ax.axis('off')
            
            # Generated color image
            gen_img = lab2rgb(gray_imgs[i], pred_ab[i])
            ax = axs[1, i]
            ax.imshow(gen_img)
            ax.set_title('Generated Color Image')
            ax.axis('off')
            
            # Real color image
            real_img = lab2rgb(gray_imgs[i], color_imgs[i])
            ax = axs[2, i]
            ax.imshow(real_img)
            ax.set_title('Real Color Image')
            ax.axis('off')
    
    plt.show()
    plt.close()