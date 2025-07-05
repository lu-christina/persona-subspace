#!/usr/bin/env python3
"""
Download feature mining data from Hugging Face using hf_hub_download.
Downloads chat_topk.h5 and pt_topk.h5 files for specified layers.
"""

import os
import pathlib
import shutil
from huggingface_hub import hf_hub_download
from typing import List


def download_layer_data(layer_id: int, base_output_dir: str = "/workspace/sae/llama-3.1-8b-instruct/feature_mining") -> bool:
    """
    Download feature mining data for a specific layer.
    
    Args:
        layer_id: Layer number (11 or 15)
        base_output_dir: Base directory for output
        
    Returns:
        True if all files downloaded successfully
    """
    print(f"📥 Downloading data for layer {layer_id}...")
    
    # Define files to download
    files_to_download = ["chat_topk.h5", "pt_topk.h5"]
    
    # Set up output directory
    output_dir = pathlib.Path(base_output_dir) / f"resid_post_layer_{layer_id}" / "trainer_1"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    success_count = 0
    
    for filename in files_to_download:
        try:
            print(f"  Downloading {filename}...")
            
            # Download file using hf_hub_download
            downloaded_path = hf_hub_download(
                repo_id="andyrdt/qwen_sae_mae",
                filename=f"resid_post_layer_{layer_id}/trainer_1/{filename}",
                repo_type="dataset",
                local_dir=None,  # Use default cache
                resume_download=True
            )
            
            # Copy to desired location
            target_path = output_dir / filename
            shutil.copy2(downloaded_path, target_path)
            
            # Get file size
            file_size = target_path.stat().st_size
            print(f"  ✅ Downloaded {filename} ({file_size:,} bytes)")
            
            success_count += 1
            
        except Exception as e:
            print(f"  ❌ Error downloading {filename}: {e}")
    
    return success_count == len(files_to_download)


def main():
    """Main function to download all required files."""
    print("🚀 Starting Llama SAE feature mining data download")
    print("=" * 60)
    
    # Layers to download
    layers_to_download = [11, 15]
    
    # Base output directory
    base_output_dir = "/workspace/sae/qwen-2.5-7b-instruct/feature_mining"
    
    # Create base directory
    pathlib.Path(base_output_dir).mkdir(parents=True, exist_ok=True)
    
    success_count = 0
    total_layers = len(layers_to_download)
    
    for layer_id in layers_to_download:
        if download_layer_data(layer_id, base_output_dir):
            success_count += 1
            print(f"✅ Layer {layer_id} download complete")
        else:
            print(f"❌ Layer {layer_id} download failed")
        print()
    
    print("=" * 60)
    print(f"📊 Download Summary: {success_count}/{total_layers} layers successful")
    
    if success_count == total_layers:
        print("🎉 All downloads completed successfully!")
        print(f"Files saved to: {base_output_dir}")
        
        # Show directory structure
        print("\n📁 Directory structure:")
        for layer_id in layers_to_download:
            layer_dir = pathlib.Path(base_output_dir) / f"resid_post_layer_{layer_id}" / "trainer_1"
            if layer_dir.exists():
                print(f"  {layer_dir}/")
                for file in layer_dir.iterdir():
                    if file.is_file():
                        size_mb = file.stat().st_size / (1024*1024)
                        print(f"    {file.name} ({size_mb:.1f} MB)")
    else:
        print("⚠️  Some downloads failed. Check the output above for details.")
    
    return success_count == total_layers


if __name__ == "__main__":
    main()