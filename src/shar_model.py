import torch
import torch.nn as nn
from .lambda_layer import LambdaLayer

class CausalConv1d(nn.Module):
    """
    1D Causal Convolution: Pads the input on the left side to ensure
    that output at time t only depends on inputs at time <= t.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, **kwargs):
        super(CausalConv1d, self).__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, **kwargs)

    def forward(self, x):
        # Pad left side only
        x = nn.functional.pad(x, (self.pad, 0))
        return self.conv(x)

class SHAREncoder(nn.Module):
    def __init__(self, in_channels=9, base_filters=64, seq_len=128):
        super(SHAREncoder, self).__init__()
        
        # 1. First 1D Causal ConvNet
        self.conv1 = CausalConv1d(in_channels, base_filters, kernel_size=8)
        self.bn1 = nn.BatchNorm1d(base_filters)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.2)
        
        # 2. Second 1D Causal ConvNet
        self.conv2 = CausalConv1d(base_filters, base_filters * 2, kernel_size=8)
        self.bn2 = nn.BatchNorm1d(base_filters * 2)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(0.2)
        
        # 3. Lambda Layer
        self.lambda_layer = LambdaLayer(
            dim_in=base_filters * 2, 
            dim_out=base_filters * 2,
            dim_k=16, 
            n_heads=4
        )
        self.bn3 = nn.BatchNorm1d(base_filters * 2)
        self.drop3 = nn.Dropout(0.2)
        
        # 4. Fully Connected (Dense) Layer mapping to representation dimension
        self.fc = nn.Linear(base_filters * 2 * seq_len, 256)
        self.relu_fc = nn.ReLU()

    def forward(self, x):
        # x shape: (Batch, Channels, SeqLen)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.drop1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.drop2(x)
        
        x = self.lambda_layer(x)
        x = self.bn3(x)
        x = self.drop3(x)
        
        # Flatten for Dense layer
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.relu_fc(x)
        
        return x

class SHAR_Pretrain(nn.Module):
    """
    Self-Supervised Pre-training model.
    Adds a Projector head to the Encoder.
    """
    def __init__(self, in_channels=9, seq_len=128, rep_dim=256, proj_dim=128):
        super(SHAR_Pretrain, self).__init__()
        self.encoder = SHAREncoder(in_channels=in_channels, seq_len=seq_len)
        
        # Projector Head
        self.projector = nn.Sequential(
            nn.Linear(rep_dim, rep_dim // 2),
            nn.BatchNorm1d(rep_dim // 2),
            nn.ReLU(),
            nn.Linear(rep_dim // 2, proj_dim)
        )

    def forward(self, x):
        h = self.encoder(x)
        z = self.projector(h)
        return z

class SHAR_Classifier(nn.Module):
    """
    Supervised Fine-Tuning model.
    Adds a Classifier head to the Encoder.
    """
    def __init__(self, encoder, rep_dim=256, num_classes=6):
        super(SHAR_Classifier, self).__init__()
        self.encoder = encoder
        
        # In fine-tuning, usually we only use the representations
        # Classifier Head
        self.classifier = nn.Sequential(
            nn.Linear(rep_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        h = self.encoder(x)
        logits = self.classifier(h)
        return logits

if __name__ == "__main__":
    b, c, l = 32, 9, 128
    x = torch.randn(b, c, l)
    
    pretrain_model = SHAR_Pretrain(in_channels=c, seq_len=l)
    out_pretrain = pretrain_model(x)
    print(f"Pre-train output shape (Projector): {out_pretrain.shape}")
    
    classifier_model = SHAR_Classifier(pretrain_model.encoder, num_classes=6)
    out_class = classifier_model(x)
    print(f"Classifier output shape: {out_class.shape}")
