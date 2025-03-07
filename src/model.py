import torch
import torch.nn as nn
from torchvision import models
import math

class CNNFeatureExtractor(nn.Module):
	def __init__(self):
		super(CNNFeatureExtractor, self).__init__()
		self.cnn_backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
		self.cnn_backbone.fc = nn.Identity()

	def forward(self, x):
		batch_size, num_frames, channels, height, width = x.shape
		x = x.view(batch_size * num_frames, channels, height, width)
		features = self.cnn_backbone(x)
		features = features.view(batch_size, num_frames, -1)
		return features

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        
        # Basic positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Learnable temporal importance
        self.temporal_importance = nn.Parameter(torch.ones(1))
        self.layer_norm = nn.LayerNorm(d_model)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        seq_len = x.size(1)
        
        # Generate temporal weights
        temporal_weights = torch.arange(seq_len, device=x.device).float() / seq_len
        temporal_weights = torch.sigmoid(temporal_weights * self.temporal_importance)
        temporal_weights = temporal_weights.unsqueeze(-1)
        
        # Apply weighted positional encoding
        pos_encoding = self.pe[:seq_len, :]
        weighted_encoding = pos_encoding * temporal_weights
        
        # Combine with input through layer norm
        x = x + weighted_encoding
        x = self.layer_norm(x)
        return self.dropout(x)

class TransformerEncoder(nn.Module):
	def __init__(self, num_heads, num_layers, d_model, output_dim, frames, dropout=0.1):
		super(TransformerEncoder, self).__init__()
		self.transformer_encoder_layer = nn.TransformerEncoderLayer(
			d_model=d_model, nhead=num_heads, dropout=dropout, batch_first=True
		)
		self.transformer_encoder = nn.TransformerEncoder(
			self.transformer_encoder_layer, num_layers=num_layers
		)
		self.out = nn.Linear(d_model, output_dim)
		self.pos_encoder = PositionalEncoding(d_model, dropout)
		self.dropout = nn.Dropout(dropout)
		self.batch_norm = nn.BatchNorm1d(frames)
		torch.nn.init.kaiming_uniform_(self.out.weight, nonlinearity='relu')

	def forward(self, x):
		output = self.batch_norm(x)
		output = self.dropout(output)
		output = self.pos_encoder(output)
		output = self.transformer_encoder(output)
		output = self.out(output)

		return output

class TransformerDecoder(nn.Module):
    def __init__(self, num_heads, num_layers, d_model, output_dim, dropout=0.1):
        super(TransformerDecoder, self).__init__()
        
        # Decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, 
            nhead=num_heads, 
            dropout=dropout, 
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, 
            num_layers=num_layers
        )
        
        # Output projection
        self.out = nn.Linear(d_model, output_dim)
        
        # Query embedding that will be learned
        self.query_embed = nn.Parameter(torch.randn(1, 8, d_model))
        
        # Normalization
        self.norm = nn.LayerNorm(d_model)
        
        # Initialization
        torch.nn.init.xavier_uniform_(self.out.weight)
    
    def forward(self, tgt, memory):
        # Expand query embeddings to batch size
        batch_size = memory.size(0)
        query_embed = self.query_embed.expand(batch_size, -1, -1)
        
        # Apply transformer decoder
        output = self.transformer_decoder(query_embed, memory)
        output = self.norm(output)
        output = self.out(output)
        
        return output

class MultiHeadClassifier(nn.Module):
    def __init__(self, d_model, num_classes):
        super(MultiHeadClassifier, self).__init__()
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        batch_size, num_frames, d_model = x.shape
        out = self.fc(x.reshape(batch_size, num_frames * d_model))
        return out

class HybridCNNTransformerModel(nn.Module):
	def __init__(
		self,
		num_classes,
		feature_dim=2048,
		frame_samples=16,
		transformer_heads=8,
		transformer_layers=4,
		transformer_outputs=64,
		decoder_layers=2,
		dropout=0.1
	):
		super(HybridCNNTransformerModel, self).__init__()
		self.cnn_feature_extractor = CNNFeatureExtractor()
		
		# Encoder
		self.transformer_encoder = TransformerEncoder(
			output_dim=feature_dim,  # Keep same dimension for encoder output 
			num_heads=transformer_heads,
			num_layers=transformer_layers,
			d_model=feature_dim,
			frames=frame_samples,
			dropout=dropout
		)
		
		# Decoder
		self.transformer_decoder = TransformerDecoder(
			num_heads=transformer_heads,
			num_layers=decoder_layers,
			d_model=feature_dim,
			output_dim=transformer_outputs,
			dropout=dropout
		)
		
		# Classification head
		self.classifier = MultiHeadClassifier(8 * transformer_outputs, num_classes)

	def forward(self, x):
		features = self.cnn_feature_extractor(x)
		encoder_output = self.transformer_encoder(features)
		
		# Process with transformer decoder
		# Using encoder output as both target and memory since we're doing self-attention
		decoder_output = self.transformer_decoder(encoder_output, encoder_output)
		out = self.classifier(decoder_output)
		
		return out