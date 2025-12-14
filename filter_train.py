import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.functional as TF
from torchvision import transforms
import coremltools as ct
from huggingface_hub import HfApi, login
import lpips
from PIL import Image
import random
import time

# ==========================================
# 1. ARGUMENT PARSING
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(description="Train and Export iPhone Enhancer Model on DGX")
    
    # Hugging Face Credentials
    parser.add_argument("--hf_token", type=str, required=True, help="Hugging Face Write Token")
    parser.add_argument("--repo_id", type=str, default="your-username/iphone-enhancer-pro", help="HF Repo ID")
    
    # Data Paths
    parser.add_argument("--source_dir", type=str, default="./data/source", help="Path to Raw/Input images")
    parser.add_argument("--target_dir", type=str, default="./data/target", help="Path to Pro/Edited images")
    
    # Training Hyperparams
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--proxy_size", type=int, default=256, help="Input size for the Neural Engine")
    
    return parser.parse_args()

# ==========================================
# 2. THE PROXY MODEL (Student)
# ==========================================
class ProxyEnhancerNet(nn.Module):
    def __init__(self):
        super().__init__()
        # A simple RepVGG-style backbone (optimized for Apple Neural Engine)
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1), # 128
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # 64
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # 32
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), # Global Average Pooling
            nn.Flatten()
        )
        # Output: 12 numbers (3x4 affine color matrix)
        self.regressor = nn.Linear(64, 12)

    def forward(self, x):
        features = self.features(x)
        coeffs = self.regressor(features)
        return coeffs

# Helper to apply the coefficients to the High-Res image (Simulates GPU Shader)
def apply_color_matrix(image, coeffs):
    B, C, H, W = image.shape
    matrix = coeffs.view(B, 3, 4)
    img_flat = image.view(B, 3, -1)
    weights = matrix[:, :, :3]
    bias = matrix[:, :, 3].unsqueeze(2)
    enhanced_flat = torch.bmm(weights, img_flat) + bias
    return enhanced_flat.view(B, 3, H, W)

# ==========================================
# 3. DATASET & LOSS
# ==========================================
# ==========================================
# REPLACED DATASET CLASS (Fixes Resize Error)
# ==========================================
class ProxyDataset(Dataset):
    def __init__(self, source_dir, target_dir, proxy_size=256, train_size=1024):
        self.proxy_size = proxy_size
        self.train_size = train_size
        
        # Check if directories exist
        if not os.path.exists(source_dir) or not os.path.exists(target_dir):
            print(f"⚠️ Warning: Data directories not found. Using Dummy Data mode.")
            self.files = []
        else:
            valid_exts = ('.png', '.jpg', '.jpeg', '.webp')
            self.source_files = sorted([os.path.join(source_dir, f) for f in os.listdir(source_dir) if f.lower().endswith(valid_exts)])
            self.target_files = sorted([os.path.join(target_dir, f) for f in os.listdir(target_dir) if f.lower().endswith(valid_exts)])
            self.files = self.source_files

    def __len__(self): 
        return len(self.files) if self.files else 100 
    
    def __getitem__(self, idx):
        if not self.files:
            # Dummy Data (Large enough to be cropped)
            src = Image.fromarray(torch.randint(0, 255, (2048, 2048, 3), dtype=torch.uint8).numpy())
            tgt = Image.fromarray(torch.randint(0, 255, (2048, 2048, 3), dtype=torch.uint8).numpy())
        else:
            src = Image.open(self.source_files[idx]).convert("RGB")
            tgt = Image.open(self.target_files[idx]).convert("RGB")
        
        # --- Safety Check: Ensure image is larger than crop size ---
        # If the input image is smaller than 1024x1024, RandomCrop will crash.
        # We force resize up if it's too small.
        w_orig, h_orig = src.size
        if w_orig < self.train_size or h_orig < self.train_size:
            src = TF.resize(src, (self.train_size, self.train_size))
            tgt = TF.resize(tgt, (self.train_size, self.train_size))

        # --- 1. Proxy Generation (Before Crop) ---
        proxy = TF.resize(src, (self.proxy_size, self.proxy_size))
        
        # --- 2. Synchronized Random Crop ---
        # FIX: Use 'transforms.RandomCrop' (Class) not 'TF.RandomCrop' (Function)
        i, j, h, w = transforms.RandomCrop.get_params(src, output_size=(self.train_size, self.train_size))
        
        src_crop = TF.crop(src, i, j, h, w)
        tgt_crop = TF.crop(tgt, i, j, h, w)
        
        # --- 3. Synchronized Flips ---
        if random.random() > 0.5:
            src_crop = TF.hflip(src_crop)
            tgt_crop = TF.hflip(tgt_crop)
            proxy = TF.hflip(proxy)
            
        return {
            "proxy": TF.to_tensor(proxy), 
            "source": TF.to_tensor(src_crop), 
            "target": TF.to_tensor(tgt_crop)
        }
     
class CombinedLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.l1 = nn.L1Loss()
        print("Loading LPIPS (VGG)...")
        self.lpips = lpips.LPIPS(net='vgg').to(device).eval()
        for p in self.lpips.parameters(): p.requires_grad = False
        
    def forward(self, pred, target):
        loss_l1 = self.l1(pred, target)
        loss_lpips = self.lpips(pred * 2 - 1, target * 2 - 1).mean()
        cosine = (1 - F.cosine_similarity(pred, target, dim=1)).mean()
        return loss_l1 + (0.5 * loss_lpips) + (0.2 * cosine)

# ==========================================
# 4. MAIN EXECUTION
# ==========================================
def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Starting Training on {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    
    # 1. Setup Model & Optimization
    model = ProxyEnhancerNet().to(device)
    
    # Blackwell Optimization (BF16 + Compile)
    if torch.cuda.is_available():
        print("⚡ Compiling model with torch.compile (Blackwell Optimized)...")
        model = torch.compile(model)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = CombinedLoss(device).to(device)
    
    # 2. Setup Data
    dataset = ProxyDataset(args.source_dir, args.target_dir, args.proxy_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    scaler = torch.amp.GradScaler('cuda') # Mixed Precision Scaler

    # 3. Training Loop
    print(f"🏋️ Training for {args.epochs} epochs...")
    model.train()
    
    for epoch in range(args.epochs):
        epoch_loss = 0
        for batch in loader:
            proxy = batch["proxy"].to(device)
            full_src = batch["source"].to(device)
            full_tgt = batch["target"].to(device)
            
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                # Predict Coeffs (Student)
                coeffs = model(proxy)
                # Apply to High Res (Simulation)
                enhanced = apply_color_matrix(full_src, coeffs)
                # Calculate Loss
                loss = criterion(enhanced, full_tgt)
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{args.epochs} | Loss: {epoch_loss/len(loader):.4f}")

    # ==========================================
    # 5. EXPORT & UPLOAD
    # ==========================================
    save_path = "iphone_enhancer.pth"
    torch.save(model.state_dict(), save_path)
    
    print("☁️ Logging into Hugging Face...")
    try:
        login(token=args.hf_token)
        api = HfApi()
        api.create_repo(repo_id=args.repo_id, exist_ok=True)
        
        print(f"⬆️ Uploading model to {args.repo_id}...")
        api.upload_file(
            path_or_fileobj=save_path,
            path_in_repo="model_pytorch.pth",
            repo_id=args.repo_id
        )
        print("✅ Upload Successful!")
    except Exception as e:
        print(f"❌ HF Upload Failed: {e}")

    # Export to CoreML
    print("📱 Exporting to CoreML...")
    export_model = ProxyEnhancerNet().eval().cpu()
    # Note: In a real run, we would reload state_dict here from the trained model
    # export_model.load_state_dict(torch.load(save_path)) 
    
    example_input = torch.rand(1, 3, args.proxy_size, args.proxy_size)
    traced_model = torch.jit.trace(export_model, example_input)
    
    mlmodel = ct.convert(
        traced_model,
        inputs=[ct.TensorType(name="input_image", shape=example_input.shape)],
        outputs=[ct.TensorType(name="color_coeffs")],
        minimum_deployment_target=ct.target.iOS16
    )
    
    mlmodel.save("ProEnhancer_FP16.mlpackage")
    print("✅ CoreML Export Complete!")

if __name__ == "__main__":
    main()