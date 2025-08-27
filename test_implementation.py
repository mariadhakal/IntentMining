#!/usr/bin/env python3
"""
Simple test script to verify the IntentMining implementation
"""

def test_imports():
    """Test if all modules can be imported"""
    try:
        from ITER_DBSCAN import ITER_DBSCAN
        print("✓ ITER_DBSCAN imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import ITER_DBSCAN: {e}")
        return False
    
    try:
        from sentenceEmbedding import SentenceEmbedding
        print("✓ SentenceEmbedding imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import SentenceEmbedding: {e}")
        return False
    
    try:
        from evaluation import EvaluateDataset
        print("✓ EvaluateDataset imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import EvaluateDataset: {e}")
        return False
    
    return True


def test_basic_functionality():
    """Test basic functionality without heavy dependencies"""
    try:
        from ITER_DBSCAN import ITER_DBSCAN
        
        # Test model initialization
        model = ITER_DBSCAN(
            initial_distance=0.3,
            initial_minimum_samples=5,
            delta_distance=0.01,
            delta_minimum_samples=1,
            max_iteration=5,
            threshold=100
        )
        print("✓ ITER_DBSCAN model initialized successfully")
        
        # Test parameter access
        assert model.initial_distance == 0.3
        assert model.initial_minimum_samples == 5
        print("✓ Model parameters set correctly")
        
        return True
        
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("Testing IntentMining Implementation")
    print("=" * 40)
    
    # Test imports
    if not test_imports():
        print("\nImport tests failed. Please check your installation.")
        return
    
    # Test basic functionality
    if not test_basic_functionality():
        print("\nBasic functionality tests failed.")
        return
    
    print("\n✓ All tests passed! The implementation is ready to use.")
    print("\nNext steps:")
    print("1. Install required packages: pip install -r requirements.txt")
    print("2. Run the example: python example.py")
    print("3. Use with your own data: python main.py --input your_data.csv")


if __name__ == "__main__":
    main()
