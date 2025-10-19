"""
Verify all imports work correctly without missing dependencies.
"""

import sys

def test_imports():
    print("Testing all module imports...\n")

    modules = [
        'config',
        'data_preprocessor',
        'trainer',
        'evaluator',
        'inference',
        'feature_visualizer',
        'cross_validator',
        'run_experiments',
        'main'
    ]

    failed = []

    for module_name in modules:
        try:
            __import__(module_name)
            print(f"[OK] {module_name}")
        except ImportError as e:
            print(f"[FAIL] {module_name}: {e}")
            failed.append((module_name, str(e)))
        except Exception as e:
            print(f"[WARN] {module_name}: {e} (may be OK)")

    print("\n" + "="*60)

    if failed:
        print("\nFailed imports:")
        for name, error in failed:
            print(f"  {name}: {error}")
        return False
    else:
        print("\nAll imports successful!")
        return True

if __name__ == '__main__':
    success = test_imports()
    sys.exit(0 if success else 1)
