from torch import nn
import torch.nn.functional as F
import torch

# Define a simple 3D convolutional autoencoder for video data
class Conv3DAutoencoder(nn.Module):
    def __init__(self, input_channels=3, base_channels=16, latent_dim=512):
        super().__init__()
        self.latent_dim = latent_dim
        self.input_channels = input_channels
        
        # Encoder: reduce spatial and temporal dimensions
        # Expected input: (B, 3, 16, 224, 224)
        self.encoder = nn.Sequential(
            # First conv layer: (B, 3, 16, 224, 224) -> (B, 16, 16, 112, 112)
            nn.Conv3d(input_channels, base_channels, kernel_size=(3,3,3), padding=1),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(),
            nn.MaxPool3d((1,2,2)),  # Reduce spatial dimensions
            
            # Second conv layer: (B, 16, 16, 112, 112) -> (B, 32, 8, 56, 56)
            nn.Conv3d(base_channels, base_channels*2, kernel_size=(3,3,3), padding=1),
            nn.BatchNorm3d(base_channels*2),
            nn.ReLU(),
            nn.MaxPool3d((2,2,2)),  # Reduce temporal and spatial dimensions
            
            # Third conv layer: (B, 32, 8, 56, 56) -> (B, 64, 4, 28, 28)
            nn.Conv3d(base_channels*2, base_channels*4, kernel_size=(3,3,3), padding=1),
            nn.BatchNorm3d(base_channels*4),
            nn.ReLU(),
            nn.MaxPool3d((2,2,2)),
            
            # Fourth conv layer: (B, 64, 4, 28, 28) -> (B, 128, 2, 14, 14)
            nn.Conv3d(base_channels*4, base_channels*8, kernel_size=(3,3,3), padding=1),
            nn.BatchNorm3d(base_channels*8),
            nn.ReLU(),
            nn.MaxPool3d((2,2,2)),
            
            # Global average pooling: (B, 128, 2, 14, 14) -> (B, 128, 1, 1, 1)
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),  # (B, 128)
            nn.Linear(base_channels*8, latent_dim),  # (B, latent_dim)
        )
        
        # Decoder: reconstruct to original size
        self.decoder_fc = nn.Linear(latent_dim, base_channels*8 * 2 * 14 * 14)
        self.decoder = nn.Sequential(
            # (B, 128, 2, 14, 14) -> (B, 64, 4, 28, 28)
            nn.ConvTranspose3d(base_channels*8, base_channels*4, kernel_size=(3,3,3), stride=(2,2,2), padding=1, output_padding=1),
            nn.BatchNorm3d(base_channels*4),
            nn.ReLU(),
            
            # (B, 64, 4, 28, 28) -> (B, 32, 8, 56, 56)
            nn.ConvTranspose3d(base_channels*4, base_channels*2, kernel_size=(3,3,3), stride=(2,2,2), padding=1, output_padding=1),
            nn.BatchNorm3d(base_channels*2),
            nn.ReLU(),
            
            # (B, 32, 8, 56, 56) -> (B, 16, 16, 112, 112)
            nn.ConvTranspose3d(base_channels*2, base_channels, kernel_size=(3,3,3), stride=(2,2,2), padding=1, output_padding=1),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(),
            
            # (B, 16, 16, 112, 112) -> (B, 3, 16, 224, 224)
            nn.ConvTranspose3d(base_channels, input_channels, kernel_size=(3,3,3), stride=(1,2,2), padding=1, output_padding=(0,1,1)),
            nn.Sigmoid(),  # assume input normalized [0,1]
        )
    
    def encode(self, x):
        """Encode input to latent space"""
        return self.encoder(x)
    
    def decode(self, z, target_shape=None):
        """Decode latent representation back to original space"""
        x = self.decoder_fc(z)
        x = x.view(x.size(0), -1, 2, 14, 14)  # (B, 128, 2, 14, 14)
        x = self.decoder(x)
        # If target_shape is provided, upsample to match input
        if target_shape is not None:
            # x: (B, 3, 16, 224, 224) or similar
            x = F.interpolate(x, size=target_shape[2:], mode='trilinear', align_corners=False)
        return x
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (B, 16, 224, 224, 3) or (B, 3, 16, 224, 224)
        
        Returns:
            Reconstructed tensor of same shape as input
        """
        # Handle different input formats
        if x.dim() == 5:
            if x.shape[1] == 16:  # (B, 16, 224, 224, 3)
                x = x.permute(0, 4, 1, 2, 3)  # -> (B, 3, 16, 224, 224)
            # If already (B, 3, 16, 224, 224), use as is
        
        original_shape = x.shape
        z = self.encode(x)
        # Pass target_shape to decoder for upsampling
        reconstruction = self.decode(z, target_shape=original_shape)
        
        # Ensure output matches input format
        reconstruction = reconstruction.permute(0, 2, 3, 4, 1)  # -> (B, 16, 224, 224, 3)
        return reconstruction