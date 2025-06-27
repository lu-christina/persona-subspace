"""
test_feature_mining.py
-----------------------
Quick test to verify the feature mining pipeline works.
"""

import sys
import torch
from feature_mining import FeatureMiningConfig, simple_text_generator, mine_top_features, load_and_display_results
from transformers import AutoTokenizer
import pathlib

def test_pipeline():
    """Test the feature mining pipeline with a small sample."""
    print("Testing feature mining pipeline...")
    
    # Small test configuration
    config = FeatureMiningConfig(
        num_samples=100,  # Very small for testing
        top_k=3,          # Few examples per feature
        batch_size=2,     # Small batch
        ctx_len=128,      # Shorter context
        out_dir=pathlib.Path("/workspace/test_feature_mining")
    )
    
    print(f"Config: {config.num_samples} samples, top_k={config.top_k}, batch_size={config.batch_size}")
    
    # Test with just a few features
    test_features = [100, 200, 300]  # Arbitrary feature indices for testing
    
    try:
        # Create text generator
        print("Creating text generator...")
        text_gen = simple_text_generator(config.num_samples)
        
        # Test the mining
        out_path = config.out_dir / "test_results.h5"
        print(f"Mining features to {out_path}...")
        
        mine_top_features(text_gen, out_path, config, specific_features=test_features)
        
        # Load and display results
        print("\nTest results:")
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        load_and_display_results(out_path, tokenizer, feature_idx=0, top_n=2)
        
        print("\n✅ Pipeline test successful!")
        return True
        
    except Exception as e:
        print(f"\n❌ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Check if CUDA is available
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Run test
    success = test_pipeline()
    
    if success:
        print("\n🎉 Feature mining setup is ready!")
        print("\nNext steps:")
        print("1. Run your get_activations.ipynb notebook to find top features")
        print("2. Update the feature indices in analyze_discovered_features.py")
        print("3. Run: python analyze_discovered_features.py --top_features '87027,45123,12345'")
    else:
        print("\n⚠️  Setup needs debugging. Check the error messages above.")
        sys.exit(1)