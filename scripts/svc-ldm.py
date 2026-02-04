import argparse
import os
import numpy as np
import torch
from tqdm import tqdm
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.models.diffusion.ddim import DDIMSampler

def load_model_from_config(config, ckpt, verbose=False):
    
    print(f"Loading audio conversion model from {ckpt}")
    
    if not ckpt.endswith(('.ckpt', '.pth', '.pt')):
        raise ValueError(f"Checkpoint file should have .ckpt, .pth or .pt extension: {ckpt}")
    
    try:
        pl_sd = torch.load(ckpt, map_location="cpu")
    except Exception as e:
        raise IOError(f"Failed to load model checkpoint from {ckpt}: {str(e)}")
    
    if "state_dict" not in pl_sd:
        raise KeyError(f"Checkpoint file {ckpt} missing 'state_dict' key")
    
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    
    if len(m) > 0 and verbose:
        print("Missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("Unexpected keys:")
        print(u)
    
    model.cuda()
    model.eval()
    return model

def main():
    
    parser = argparse.ArgumentParser(description="Voice Timbre Conversion for Unconditional Model")
    
    
    parser.add_argument("--source_feature", type=str, required=True,
                        help="Path to source timbre feature file (.npy)")
    parser.add_argument("--outdir", type=str, default="conversion_results",
                        help="Output directory for converted features")
    
    
    parser.add_argument("--config", type=str, required=True,
                        help="Path to model config file (.yaml)")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to trained model checkpoint (.ckpt)")
    
    
    parser.add_argument("--ddim_steps", type=int, default=100,
                        help="Number of diffusion steps for conversion")
    parser.add_argument("--chunk_size", type=int, default=128,
                        help="Number of frames per chunk for long sequences")
    parser.add_argument("--output_prefix", type=str, default="converted",
                        help="Prefix for output feature files")
    
    opt = parser.parse_args()
    
    
    os.makedirs(opt.outdir, exist_ok=True)
    
    
    config = OmegaConf.load(opt.config)
    
    
    model = load_model_from_config(config, opt.ckpt)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    sampler = DDIMSampler(model)
    
    
    print(f"Loading source features from {opt.source_feature}...")
    source_features = np.load(opt.source_feature)
    
    num_frames = source_features.shape[1]
    
    
    
    chunks = []
    for start in range(0, num_frames, opt.chunk_size):
        end = min(start + opt.chunk_size, num_frames)
        chunks.append((start, end))
    
    converted_features = []
    
    
    print(f"Converting timbre using unconditional model...")
    
    with torch.no_grad():
        for idx, (start, end) in tqdm(enumerate(chunks), total=len(chunks), desc="Processing"):
           
            chunk_length = end - start
            chunk_features = source_features[:, start:end]
            
            chunk_tensor = torch.from_numpy(chunk_features).float()
            chunk_tensor = chunk_tensor.unsqueeze(0).to(device)  # (1, 1, chunk_length)
            
            shape = (config.model.params.channels, config.model.params.audio_size)
            
            conditioning = None
            unconditional_conditioning = None
            
            samples, _ = sampler.sample(
                S=opt.ddim_steps,
                conditioning=conditioning,
                batch_size=1,
                shape=shape,
                verbose=False,
                unconditional_guidance_scale=1.0,  
                unconditional_conditioning=unconditional_conditioning,
                eta=0.0,
                x_T=chunk_tensor  
            )
            
            converted_chunk = model.decode_first_stage(samples)
            converted_features.append(converted_chunk.squeeze(0).cpu().numpy())
    
    converted_full = np.concatenate(converted_features, axis=1)

    converted_full = np.ascontiguousarray(converted_full, dtype=np.float32)

    converted_full = converted_full.flatten()  
    print(f"Final output shape: {converted_full.shape}")  

    base_name = os.path.basename(opt.source_feature).split('.')[0]
    output_path = os.path.join(opt.outdir, f"{opt.output_prefix}_{base_name}.npy")

    try:
        np.save(output_path, converted_full, allow_pickle=False)  # 禁用pickle确保标准格式
        print(f"Converted features saved to: {output_path}")
        print(f"Array shape: {converted_full.shape}, dtype: {converted_full.dtype}")
    except Exception as e:
        print(f"Failed to save .npy file: {str(e)}")

        try:
            with open(output_path, 'wb') as f:
                np.lib.format.write_array(f, converted_full, allow_pickle=False)
            print(f"Used alternative method to save .npy file")
        except Exception as e2:
            print(f"Completely failed to save file: {str(e2)}")
            raise


if __name__ == "__main__":
    main()
