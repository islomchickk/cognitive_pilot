from pathlib import Path
import matplotlib.pyplot as plt

def visualize_rotation(original_image, rotated_image, angle, epoch, config):
    """
    Visualize original and rotated RGB images side by side
    
    Args:
        original_image (torch.Tensor): Original image tensor (N,C,H,W)
        rotated_image (torch.Tensor): Rotated image tensor (N,C,H,W)
        angle (float): Rotation angle in degrees
        epoch (int): Current training epoch
    """
    save_folder = Path(config.training.save_folder, "visualizations")
    # Convert from tensor (C,H,W) to numpy array (H,W,C) and move channels to last dimension
    original = original_image.cpu().squeeze().permute(1,2,0).detach().numpy()
    rotated = rotated_image.cpu().squeeze().permute(1,2,0).detach().numpy()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    ax1.imshow(original)
    ax1.set_title('Original')
    ax1.axis('off')
    
    ax2.imshow(rotated)
    ax2.set_title(f'Rotated by {angle:.1f}°')
    ax2.axis('off')
    
    plt.suptitle(f'Epoch {epoch}')
    plt.tight_layout()
    plt.savefig(f'{save_folder}/rotation_vis_epoch_{epoch}.png')
    plt.close()