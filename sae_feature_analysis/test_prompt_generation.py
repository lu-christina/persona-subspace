#!/usr/bin/env python3
"""Test script for generate_entity_behavior_semantic_prompt function"""

import os
import pathlib
import h5py
import pandas as pd
import asyncio
from entity_behavior_semantic_autointerp import generate_entity_behavior_semantic_prompt, analyze_features_with_claude

def test_h5_files_exist():
    """Check if the expected .h5 files exist"""
    print("🔍 Checking if .h5 files exist...")
    
    # Test paths for both models
    test_configs = [
        {
            "model_name": "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "model_dir": "llama-3.1-8b-instruct",
            "layer": 15,
            "trainer": 0
        },
        {
            "model_name": "Qwen/Qwen2.5-7B-Instruct", 
            "model_dir": "qwen-2.5-7b-instruct",
            "layer": 15,
            "trainer": 1
        }
    ]
    
    for config in test_configs:
        base_path = f"/workspace/sae/{config['model_dir']}/feature_mining/resid_post_layer_{config['layer']}/trainer_{config['trainer']}"
        chat_file = pathlib.Path(base_path) / "chat_topk.h5"
        pretrain_file = pathlib.Path(base_path) / "pt_topk.h5"
        
        print(f"\n📁 Testing {config['model_name']}:")
        print(f"  Chat file: {chat_file}")
        print(f"  Exists: {chat_file.exists()}")
        print(f"  Pretrain file: {pretrain_file}")
        print(f"  Exists: {pretrain_file.exists()}")
        
        # Try to peek inside if files exist
        if chat_file.exists():
            try:
                with h5py.File(chat_file, 'r') as f:
                    print(f"  Chat file keys: {list(f.keys())}")
                    if 'scores' in f:
                        print(f"  Chat scores shape: {f['scores'].shape}")
            except Exception as e:
                print(f"  Error reading chat file: {e}")
        
        if pretrain_file.exists():
            try:
                with h5py.File(pretrain_file, 'r') as f:
                    print(f"  Pretrain file keys: {list(f.keys())}")
                    if 'scores' in f:
                        print(f"  Pretrain scores shape: {f['scores'].shape}")
            except Exception as e:
                print(f"  Error reading pretrain file: {e}")

def test_prompt_generation():
    """Test the prompt generation function"""
    print("\n🧪 Testing prompt generation...")
    
    # Test with Llama model
    try:
        print("\n📝 Testing with Llama model...")
        prompt = generate_entity_behavior_semantic_prompt(
            layer=15,
            trainer=0, 
            feature_id=100,
            model_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
            num_chat_examples=2,
            num_pretrain_examples=2
        )
        print(f"✅ Llama prompt generated successfully (length: {len(prompt)} chars)")
        print(f"First 500 chars:\n{prompt}")
        
    except Exception as e:
        print(f"❌ Error with Llama model: {e}")
    
    # Test with Qwen model
    try:
        print("\n📝 Testing with Qwen model...")
        prompt = generate_entity_behavior_semantic_prompt(
            layer=15,
            trainer=1,
            feature_id=100, 
            model_name="Qwen/Qwen2.5-7B-Instruct",
            num_chat_examples=2,
            num_pretrain_examples=2
        )
        print(f"✅ Qwen prompt generated successfully (length: {len(prompt)} chars)")
        print(f"First 500 chars:\n{prompt}")
        
    except Exception as e:
        print(f"❌ Error with Qwen model: {e}")

def test_custom_path():
    """Test with custom path template"""
    print("\n🔧 Testing with custom path template...")
    
    try:
        # Test with explicit path template
        custom_template = "/workspace/sae/llama-3.1-8b-instruct/feature_mining/resid_post_layer_15/trainer_0"
        prompt = generate_entity_behavior_semantic_prompt(
            layer=15,
            trainer=0,
            feature_id=50,
            feature_mining_path_template=custom_template,
            model_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
            num_chat_examples=1,
            num_pretrain_examples=1
        )
        print(f"✅ Custom path template worked (length: {len(prompt)} chars)")
        
    except Exception as e:
        print(f"❌ Error with custom path: {e}")

async def test_claude_analysis():
    """Test full Claude analysis with first feature from CSV"""
    print("\n🤖 Testing full Claude analysis...")
    
    try:
        # Load the CSV file to get the first feature
        csv_path = "/root/git/persona-subspace/sae_feature_analysis/results/universal/universal_30.csv"
        df = pd.read_csv(csv_path)
        
        if len(df) == 0:
            print("❌ No features found in CSV")
            return
        
        # Get the first feature
        first_row = df.iloc[0]
        source = first_row['source']  # e.g., "llama_trainer1_layer11_asst"
        feature_id = int(first_row['feature_id'])
        
        # Parse the source to extract model, layer, trainer info
        parts = source.split('_')
        if len(parts) < 4:
            print(f"❌ Could not parse source: {source}")
            return
        
        model_name = parts[0]  # "llama" or "qwen"
        trainer = int(parts[1].replace('trainer', ''))
        layer = int(parts[2].replace('layer', ''))
        
        # Map to full model names
        if model_name == "llama":
            full_model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        elif model_name == "qwen": 
            full_model_name = "Qwen/Qwen2.5-7B-Instruct"
        else:
            print(f"❌ Unknown model: {model_name}")
            return
        
        print(f"📊 Analyzing feature {feature_id} from {source}")
        print(f"   Model: {full_model_name}")
        print(f"   Layer: {layer}, Trainer: {trainer}")
        
        # Run Claude analysis
        results = await analyze_features_with_claude(
            feature_ids=[feature_id],
            layer=layer,
            trainer=trainer,
            max_concurrent=1,
            output_dir="/root/git/persona-subspace/sae_feature_analysis/.cache",
            feature_mining_path_template=None,  # Let it use defaults
            claude_model="claude-3-5-sonnet-20241022",
            baseline_model_path=full_model_name
        )
        
        if feature_id in results:
            result = results[feature_id]
            print(f"\n✅ Claude analysis completed!")
            print(f"📝 Feature Description: {result['feature_description']}")
            print(f"🏷️  Feature Type: {result['type']}")
            print(f"\n🤖 Full Claude Response:")
            print("="*50)
            print(result['claude_completion'])
            print("="*50)
        else:
            print(f"❌ No results returned for feature {feature_id}")
            
    except Exception as e:
        print(f"❌ Error in Claude analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🚀 Starting prompt generation tests...")
    test_h5_files_exist()
    test_prompt_generation()
    test_custom_path()
    
    # Run Claude analysis test
    print("\n" + "="*60)
    print("🤖 CLAUDE ANALYSIS TEST")
    print("="*60)
    asyncio.run(test_claude_analysis())
    
    print("\n✅ All tests completed!")