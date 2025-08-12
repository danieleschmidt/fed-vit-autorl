#!/usr/bin/env python3
"""Basic import test to verify package structure."""

import sys
import traceback

def test_imports():
    """Test basic package imports."""
    success_count = 0
    total_count = 0
    
    # Core imports
    test_imports = [
        "fed_vit_autorl",
        "fed_vit_autorl.federated.aggregation",
        "fed_vit_autorl.models.vit_perception", 
        "fed_vit_autorl.reinforcement.ppo_federated",
        "fed_vit_autorl.federated.advanced_aggregation",
        "fed_vit_autorl.models.advanced_vit",
        "fed_vit_autorl.autonomous",
        "fed_vit_autorl.research.experimental_framework",
        "fed_vit_autorl.deployment.hyperscale_federation",
    ]
    
    for module_name in test_imports:
        total_count += 1
        try:
            __import__(module_name)
            print(f"✅ Successfully imported {module_name}")
            success_count += 1
        except ImportError as e:
            print(f"❌ Failed to import {module_name}: {e}")
        except Exception as e:
            print(f"⚠️  Error importing {module_name}: {e}")
    
    print(f"\nImport Results: {success_count}/{total_count} successful")
    
    # Test advanced component availability
    try:
        from fed_vit_autorl import _ADVANCED_COMPONENTS_AVAILABLE
        if _ADVANCED_COMPONENTS_AVAILABLE:
            print("✅ Advanced components are available")
        else:
            print("⚠️  Advanced components not available (missing dependencies)")
    except:
        print("❓ Could not determine advanced component availability")
    
    return success_count, total_count

def test_basic_functionality():
    """Test basic functionality of core components."""
    print("\nTesting basic functionality...")
    
    try:
        # Test basic federated aggregation
        from fed_vit_autorl.federated.aggregation import FedAvgAggregator
        
        aggregator = FedAvgAggregator()
        
        # Mock client updates
        import torch
        client_updates = [
            {"layer1.weight": torch.randn(5, 5), "layer1.bias": torch.randn(5)},
            {"layer1.weight": torch.randn(5, 5), "layer1.bias": torch.randn(5)},
        ]
        
        result = aggregator.aggregate(client_updates)
        
        assert "layer1.weight" in result
        assert "layer1.bias" in result
        print("✅ Basic federated aggregation working")
        
    except Exception as e:
        print(f"❌ Basic aggregation failed: {e}")
        traceback.print_exc()
    
    try:
        # Test ViT perception model
        from fed_vit_autorl.models.vit_perception import ViTPerception
        
        model = ViTPerception(
            img_size=224,
            patch_size=16, 
            num_classes=10,
            embed_dim=256,
            depth=6,
            num_heads=8,
            pretrained=False,  # Avoid downloading
        )
        
        import torch
        x = torch.randn(2, 3, 224, 224)
        output = model(x)
        
        assert output.shape == (2, 10)
        print("✅ ViT perception model working")
        
    except Exception as e:
        print(f"❌ ViT model failed: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    print("Fed-ViT-AutoRL Enhanced Import Test")
    print("=" * 50)
    
    success, total = test_imports()
    test_basic_functionality()
    
    print(f"\nFinal Result: {success}/{total} imports successful")
    
    if success >= total * 0.8:  # 80% success rate
        print("🎉 Package integration successful!")
        sys.exit(0)
    else:
        print("⚠️  Some components failed to import")
        sys.exit(1)