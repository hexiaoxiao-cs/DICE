"""
Minimal example demonstrating DICE concepts

This script shows the core DICE algorithm without requiring model downloads.
It uses random tensors to demonstrate the inversion and editing pipeline.
"""

import torch
from dice import invert_new, sample_ddpm_inverse_new, generate_mask_schedule

def main():
    print("🎲 DICE - Minimal Example")
    print("=" * 50)
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Simulate a simple discrete diffusion model
    class SimpleModel(torch.nn.Module):
        def __init__(self, num_labels=8192):
            super().__init__()
            self.num_labels = num_labels
            self.linear = torch.nn.Linear(num_labels, num_labels)
        
        def forward(self, x, t, **kwargs):
            # Simple linear transformation (for demo purposes)
            x_onehot = torch.nn.functional.one_hot(x, self.num_labels).float()
            x_onehot = x_onehot.permute(0, 3, 1, 2)  # [B, C, H, W]
            logits = self.linear(x_onehot.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            return logits
        
        def add_noise(self, x, t, random_x=None):
            if random_x is None:
                random_x = torch.randint(0, self.num_labels, x.shape, device=x.device)
            mask = (torch.rand_like(x.float()) <= t[:, None, None]).long()
            return x * (1 - mask) + random_x * mask, mask
    
    model = SimpleModel().to(device)
    model.eval()
    
    # Create a "clean" image (random tokens)
    latent_shape = (1, 8, 8)  # Small size for demo
    x0 = torch.randint(0, 8192, latent_shape, device=device)
    print(f"✓ Created clean tokens: shape {x0.shape}")
    
    # Simulate conditioning (random for demo)
    model_inputs = {"byt5": torch.randn(1, 256, device=device)}
    unconditional_inputs = {"byt5": torch.zeros(1, 256, device=device)}
    
    # Step 1: Inversion
    print("\n📍 Step 1: Inversion (recording noise sequence)")
    with torch.no_grad():
        final_noisy, intermediates, zs, mask_schedule, Tts = invert_new(
            model=model,
            x0=x0,
            model_inputs=model_inputs,
            latent_shape=latent_shape,
            cfg=(8.0, 8.0),
            t_start=0.0,
            t_end=0.75,
            steps=12,
            device=device
        )
    
    print(f"✓ Recorded {len(zs)} noise vectors")
    print(f"✓ Generated {len(mask_schedule)} masks")
    print(f"✓ Final noisy tokens shape: {final_noisy.shape}")
    
    # Step 2: Sampling/Editing
    print("\n📍 Step 2: Sampling (using recorded noise)")
    init_noise = torch.randint(0, 8192, latent_shape, device=device)
    
    with torch.no_grad():
        edited, _ = sample_ddpm_inverse_new(
            model=model,
            model_inputs=model_inputs,
            latent_shape=latent_shape,
            zs=zs[::-1],  # Reverse order for sampling
            mask_schedule=mask_schedule[::-1],
            init_noise=init_noise,
            init_x=final_noisy.long(),
            unconditional_inputs=unconditional_inputs,
            cfg=(8.0, 8.0),
            lamb=0.7,
            t_start=0.75,
            t_end=0.0,
            steps=12,
            device=device
        )
    
    print(f"✓ Generated edited tokens: shape {edited.shape}")
    
    # Compute similarity
    match_ratio = (edited == x0).float().mean().item()
    print(f"\n📊 Token match ratio: {match_ratio:.2%}")
    
    print("\n✅ Demo completed successfully!")
    print("\n💡 Next steps:")
    print("   - Download model weights from the README")
    print("   - Run the full examples in notebooks/")
    print("   - Check configs/ for hyperparameter settings")

if __name__ == "__main__":
    main()
