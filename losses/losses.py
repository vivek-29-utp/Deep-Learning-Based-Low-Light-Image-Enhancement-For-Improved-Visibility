

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class L_color(nn.Module):
    """
    Color Constancy Loss - Academic Implementation
    
    Maintains color consistency by ensuring the average values of
    RGB channels are similar to prevent color casting.
    
    This is an academic implementation for educational purposes.
    """

    def __init__(self):
        super(L_color, self).__init__()

    def forward(self, x):
        """
        Calculate color constancy loss
        
        Args:
            x (torch.Tensor): Enhanced image [B, 3, H, W]
            
        Returns:
            torch.Tensor: Color constancy loss
        """
        b, c, h, w = x.shape

        mean_rgb = torch.mean(x, [2, 3], keepdim=True)
        mr, mg, mb = torch.split(mean_rgb, 1, dim=1)
        Drg = torch.pow(mr - mg, 2)
        Drb = torch.pow(mr - mb, 2)
        Dgb = torch.pow(mb - mg, 2)
        k = torch.pow(torch.pow(Drg, 2) + torch.pow(Drb, 2) + torch.pow(Dgb, 2), 0.5)

        return k


class L_spa(nn.Module):
    """
    Spatial Consistency Loss - Academic Implementation
    
    Ensures local spatial consistency by measuring the difference
    between neighboring pixels in the enhanced image.
    
    This is an academic implementation for educational purposes.
    """

    def __init__(self):
        super(L_spa, self).__init__()
        # Fixed kernels for spatial consistency (academic implementation)
        kernel_left = torch.FloatTensor([[0, 0, 0], [-1, 1, 0], [0, 0, 0]]).unsqueeze(0).unsqueeze(0)
        kernel_right = torch.FloatTensor([[0, 0, 0], [0, 1, -1], [0, 0, 0]]).unsqueeze(0).unsqueeze(0)
        kernel_up = torch.FloatTensor([[0, -1, 0], [0, 1, 0], [0, 0, 0]]).unsqueeze(0).unsqueeze(0)
        kernel_down = torch.FloatTensor([[0, 0, 0], [0, 1, 0], [0, -1, 0]]).unsqueeze(0).unsqueeze(0)
        
        self.weight_left = nn.Parameter(data=kernel_left, requires_grad=False)
        self.weight_right = nn.Parameter(data=kernel_right, requires_grad=False)
        self.weight_up = nn.Parameter(data=kernel_up, requires_grad=False)
        self.weight_down = nn.Parameter(data=kernel_down, requires_grad=False)
        self.pool = nn.AvgPool2d(4)
        
    def forward(self, org, enhance):
        """
        Calculate spatial consistency loss
        
        Args:
            org (torch.Tensor): Original image [B, 3, H, W]
            enhance (torch.Tensor): Enhanced image [B, 3, H, W]
            
        Returns:
            torch.Tensor: Spatial consistency loss
        """
        b, c, h, w = org.shape

        org_mean = torch.mean(org, 1, keepdim=True)
        enhance_mean = torch.mean(enhance, 1, keepdim=True)

        org_pool = self.pool(org_mean)			
        enhance_pool = self.pool(enhance_mean)	

        # Weighting mechanism (academic implementation)
        weight_diff = torch.max(torch.FloatTensor([1]).to(org.device) + 10000 * torch.min(org_pool - torch.FloatTensor([0.3]).to(org.device), torch.FloatTensor([0]).to(org.device)), torch.FloatTensor([0.5]).to(org.device))
        E_1 = torch.mul(torch.sign(enhance_pool - torch.FloatTensor([0.5]).to(org.device)), enhance_pool - org_pool)

        # Calculate differences in four directions
        D_org_left = F.conv2d(org_pool, self.weight_left, padding=1)
        D_org_right = F.conv2d(org_pool, self.weight_right, padding=1)
        D_org_up = F.conv2d(org_pool, self.weight_up, padding=1)
        D_org_down = F.conv2d(org_pool, self.weight_down, padding=1)

        D_enhance_left = F.conv2d(enhance_pool, self.weight_left, padding=1)
        D_enhance_right = F.conv2d(enhance_pool, self.weight_right, padding=1)
        D_enhance_up = F.conv2d(enhance_pool, self.weight_up, padding=1)
        D_enhance_down = F.conv2d(enhance_pool, self.weight_down, padding=1)

        D_left = torch.pow(D_org_left - D_enhance_left, 2)
        D_right = torch.pow(D_org_right - D_enhance_right, 2)
        D_up = torch.pow(D_org_up - D_enhance_up, 2)
        D_down = torch.pow(D_org_down - D_enhance_down, 2)
        E = (D_left + D_right + D_up + D_down)

        return E


class L_exp(nn.Module):
    """
    Exposure Control Loss - Academic Implementation
    
    Ensures the enhanced image has proper exposure by measuring the
    intensity of well-exposed pixels.
    
    This is an academic implementation for educational purposes.
    """

    def __init__(self, patch_size):
        super(L_exp, self).__init__()
        self.pool = nn.AvgPool2d(patch_size)
        
    def forward(self, x, mean_val):
        """
        Calculate exposure control loss
        
        Args:
            x (torch.Tensor): Enhanced image [B, 3, H, W]
            mean_val (float): Target exposure value (typically 0.6)
            
        Returns:
            torch.Tensor: Exposure control loss
        """
        b, c, h, w = x.shape
        x = torch.mean(x, 1, keepdim=True)
        mean = self.pool(x)

        d = torch.mean(torch.pow(mean - torch.FloatTensor([mean_val]).to(x.device), 2))
        return d


class L_TV(nn.Module):
    """
    Illumination Smoothness Loss - Academic Implementation
    
    Promotes smooth illumination curves by penalizing sharp changes
    using Total Variation loss.
    
    This is an academic implementation for educational purposes.
    """
    def __init__(self, TVLoss_weight=1):
        super(L_TV, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        """
        Calculate total variation loss
        
        Args:
            x (torch.Tensor): Enhancement curves [B, 3, H, W]
            
        Returns:
            torch.Tensor: Total variation loss
        """
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = (x.size()[2] - 1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size


class ZeroDCELoss(nn.Module):
    """
    Complete Zero-DCE++ Loss Function - Academic Implementation
    
    Combines all individual loss components with academic weights.
    This academic implementation follows the loss configuration.
    
    Academic Loss Weights:
    - L_TV: 1600 (illumination smoothness)
    - L_spa: 1.0 (spatial consistency)
    - L_color: 5.0 (color constancy)
    - L_exp: 10.0 (exposure control)
    """
    
    def __init__(self, 
                 tv_weight=1200.0,  # Reduced from 1600 for less over-smoothing
                 spa_weight=1.5,    # Increased from 1.0 for better spatial consistency
                 color_weight=3.0,   # Reduced from 5.0 for more natural colors
                 exp_weight=15.0,    # Increased from 10.0 for better exposure control
                 patch_size=16,
                 mean_val=0.7):      # Increased from 0.6 for brighter target
        """
        Initialize complete Zero-DCE++ loss with academic weights
        
        Args:
            tv_weight (float): Weight for illumination smoothness loss
            spa_weight (float): Weight for spatial consistency loss
            color_weight (float): Weight for color constancy loss
            exp_weight (float): Weight for exposure control loss
            patch_size (int): Patch size for exposure loss
            mean_val (float): Target exposure value
        """
        super(ZeroDCELoss, self).__init__()
        
        # Initialize individual loss components (academic implementation)
        self.L_color = L_color()
        self.L_spa = L_spa()
        self.L_exp = L_exp(patch_size)
        self.L_TV = L_TV()
        
        # Academic loss weights
        self.tv_weight = tv_weight
        self.spa_weight = spa_weight
        self.color_weight = color_weight
        self.exp_weight = exp_weight
        self.mean_val = mean_val
        
    def forward(self, original_image, enhanced_image, curves):
        """
        Calculate total Zero-DCE++ loss - academic implementation
        
        Args:
            original_image (torch.Tensor): Original low-light image [B, 3, H, W]
            enhanced_image (torch.Tensor): Enhanced image [B, 3, H, W] 
            curves (torch.Tensor): Predicted illumination curves [B, 3, H, W]
            
        Returns:
            dict: Dictionary containing individual and total losses
        """
        # Calculate individual losses (academic implementation)
        loss_tv = self.tv_weight * self.L_TV(curves)
        loss_spa = self.spa_weight * torch.mean(self.L_spa(enhanced_image, original_image))
        loss_col = self.color_weight * torch.mean(self.L_color(enhanced_image))
        loss_exp = self.exp_weight * torch.mean(self.L_exp(enhanced_image, self.mean_val))
        
        # Total loss (academic formula)
        total_loss = loss_tv + loss_spa + loss_col + loss_exp
        
        # Return all losses for monitoring
        return {
            'total_loss': total_loss,
            'smoothness_loss': loss_tv / self.tv_weight,  # Unweighted for monitoring
            'spatial_loss': loss_spa / self.spa_weight,
            'color_loss': loss_col / self.color_weight,
            'exposure_loss': loss_exp / self.exp_weight,
            # Also return weighted losses for reference
            'weighted_smoothness': loss_tv,
            'weighted_spatial': loss_spa,
            'weighted_color': loss_col,
            'weighted_exposure': loss_exp
        }


def test_losses():
    """
    Test function to verify all loss components work correctly
    """
    print("Testing Zero-DCE++ Academic Loss Functions...")
    
    # Create dummy data
    batch_size = 2
    height, width = 256, 256
    
    original = torch.rand(batch_size, 3, height, width)
    enhanced = torch.rand(batch_size, 3, height, width)
    curves = torch.rand(batch_size, 3, height, width)
    
    # Test individual losses
    color_loss_fn = L_color()
    spa_loss_fn = L_spa()
    exp_loss_fn = L_exp(16)
    tv_loss_fn = L_TV()
    
    color_loss = color_loss_fn(enhanced)
    spa_loss = spa_loss_fn(original, enhanced)
    exp_loss = exp_loss_fn(enhanced, 0.6)
    tv_loss = tv_loss_fn(curves)
    
    print(f"Color Loss: {color_loss.item():.6f}")
    print(f"Spatial Loss: {spa_loss.item():.6f}")
    print(f"Exposure Loss: {exp_loss.item():.6f}")
    print(f"TV Loss: {tv_loss.item():.6f}")
    
    # Test complete loss
    loss_fn = ZeroDCELoss()
    losses = loss_fn(original, enhanced, curves)
    
    print(f"\nTotal Loss: {losses['total_loss'].item():.6f}")
    print(f"Weighted components: "
          f"TV={losses['weighted_smoothness']:.6f} + "
          f"Spatial={losses['weighted_spatial']:.6f} + "
          f"Color={losses['weighted_color']:.6f} + "
          f"Exp={losses['weighted_exposure']:.6f}")
    
    print("Academic loss functions test completed successfully!")


if __name__ == "__main__":
    test_losses()
