

import torch
import torch.nn as nn
import torch.nn.functional as F


class CSDN_Tem(nn.Module):
    """
    Channel-wise and Spatial-wise Depth Network (CSDN)
    Depthwise separable convolution used in academic implementation
    
    This is an academic implementation for educational purposes.
    """
    def __init__(self, in_ch, out_ch):
        super(CSDN_Tem, self).__init__()
        # Depthwise convolution
        self.depth_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=in_ch
        )
        # Pointwise convolution
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1
        )

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out


class ZeroDCEPP(nn.Module):
    """
    Zero-DCE++ Model - Academic Architecture
    
    This academic implementation follows the enhance_net_nopool architecture.
    The model predicts illumination enhancement curves A and applies them
    iteratively to enhance low-light images.
    
    Architecture:
    - 7 CSDN_Tem layers with skip connections
    - Iterative enhancement function (8 iterations)
    - Conservative enhancement philosophy
    """
    
    def __init__(self, scale_factor=1):
        super(ZeroDCEPP, self).__init__()
        
        self.relu = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=self.scale_factor)
        number_f = 32

        # Zero-DCE++ DWC + point-wise shared layers (academic implementation)
        self.e_conv1 = CSDN_Tem(3, number_f) 
        self.e_conv2 = CSDN_Tem(number_f, number_f) 
        self.e_conv3 = CSDN_Tem(number_f, number_f) 
        self.e_conv4 = CSDN_Tem(number_f, number_f) 
        self.e_conv5 = CSDN_Tem(number_f * 2, number_f) 
        self.e_conv6 = CSDN_Tem(number_f * 2, number_f) 
        self.e_conv7 = CSDN_Tem(number_f * 2, 3) 
        
        # Initialize weights using academic method
        self._initialize_weights()

    def enhance(self, x, x_r):
        """
        Academic Zero-DCE++ enhancement function
        
        Applies the learned illumination curves iteratively (8 times)
        using the formula: x = x + x_r * (x^2 - x)
        
        Args:
            x (torch.Tensor): Input image
            x_r (torch.Tensor): Learned enhancement curves
            
        Returns:
            torch.Tensor: Enhanced image
        """
        # Apply enhancement iteratively (academic implementation)
        x = x + x_r * (torch.pow(x, 2) - x)
        x = x + x_r * (torch.pow(x, 2) - x)
        x = x + x_r * (torch.pow(x, 2) - x)
        enhance_image_1 = x + x_r * (torch.pow(x, 2) - x)        
        x = enhance_image_1 + x_r * (torch.pow(enhance_image_1, 2) - enhance_image_1)        
        x = x + x_r * (torch.pow(x, 2) - x)    
        x = x + x_r * (torch.pow(x, 2) - x)
        enhance_image = x + x_r * (torch.pow(x, 2) - x)    

        return enhance_image
        
    def forward(self, x):
        """
        Forward pass - academic implementation
        
        Args:
            x (torch.Tensor): Input low-light image [B, 3, H, W] in [0,1]
            
        Returns:
            tuple: (enhanced_image, enhancement_curves)
        """
        # Handle scaling if needed
        if self.scale_factor == 1:
            x_down = x
        else:
            x_down = F.interpolate(x, scale_factor=1/self.scale_factor, mode='bilinear')

        # Feature extraction with skip connections (academic implementation)
        x1 = self.relu(self.e_conv1(x_down))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))
        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))
        x_r = F.tanh(self.e_conv7(torch.cat([x1, x6], 1)))
        
        # Handle scaling for enhancement curves
        if self.scale_factor == 1:
            x_r = x_r
        else:
            x_r = self.upsample(x_r)
            
        # Apply enhancement
        enhance_image = self.enhance(x, x_r)
        
        return enhance_image, x_r
    
    def _initialize_weights(self):
        """
        Initialize weights using academic Zero-DCE++ method
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Academic initialization: normal distribution with std=0.02
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()


def create_model(device='cpu', scale_factor=1):
    """
    Helper function to create and initialize Zero-DCE++ model
    
    Args:
        device (str): Device to place model on ('cpu' or 'cuda')
        scale_factor (int): Scale factor for input images (default: 1)
        
    Returns:
        ZeroDCEPP: Initialized Zero-DCE++ model
    """
    model = ZeroDCEPP(scale_factor=scale_factor).to(device)
    return model


if __name__ == "__main__":
    # Test the model
    print("Testing Zero-DCE++ Academic Model...")
    
    # Create model
    model = create_model()
    
    # Test with dummy input
    test_input = torch.randn(1, 3, 256, 256)
    
    # Forward pass
    enhanced, curves = model(test_input)
    
    print(f"Input shape: {test_input.shape}")
    print(f"Enhanced shape: {enhanced.shape}")
    print(f"Curves shape: {curves.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Verify output ranges
    print(f"Enhanced range: [{enhanced.min().item():.3f}, {enhanced.max().item():.3f}]")
    print(f"Curves range: [{curves.min().item():.3f}, {curves.max().item():.3f}]")
    
    print("Zero-DCE++ Academic Model test completed successfully!")
