#!/usr/bin/env python3

import os
import sys
import subprocess
from pathlib import Path

def run_command(cmd, description):
    print(f"\n{'='*70}")
    print(f"{description}")
    print(f"{'='*70}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"⚠ Warning: Command failed with code {result.returncode}")
    return result.returncode == 0

def main():
    print("""
╔════════════════════════════════════════════════════════════════╗
║                                                                ║
║         KERNELMIND Quick Start Setup & Verification            ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝
    """)
    
    kernelmind_path = Path(__file__).parent
    os.chdir(kernelmind_path)
    
    print(f"Working directory: {kernelmind_path}\n")
    
    print("Step 1: Checking Python version...")
    python_version = sys.version_info
    if python_version.major >= 3 and python_version.minor >= 10:
        print(f"✓ Python {python_version.major}.{python_version.minor} OK")
    else:
        print(f"✗ Python 3.10+ required (you have {python_version.major}.{python_version.minor})")
        return False
    
    print("\nStep 2: Checking virtual environment...")
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✓ Virtual environment active")
    else:
        print("⚠ Not in virtual environment. Recommended: source venv/bin/activate")
    
    print("\nStep 3: Checking required packages...")
    packages_ok = True
    required_packages = ['torch', 'numpy', 'anthropic']
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} installed")
        except ImportError:
            print(f"✗ {package} not found")
            packages_ok = False
    
    if not packages_ok:
        print("\nInstalling missing packages...")
        run_command("pip install -r requirements.txt", "Installing dependencies")
    
    print("\nStep 4: Checking API key...")
    if os.getenv("ANTHROPIC_API_KEY"):
        print("✓ ANTHROPIC_API_KEY set in environment")
    elif Path(".env").exists():
        print("✓ .env file found")
    else:
        print("⚠ ANTHROPIC_API_KEY not configured")
        print("  Please create .env file with: ANTHROPIC_API_KEY=your_key_here")
        print("  Or run: export ANTHROPIC_API_KEY=your_key_here")
    
    print("\nStep 5: Verifying imports...")
    try:
        from kernelmind.core import ModelParser
        from kernelmind.agent import OptimizationAgent
        from kernelmind.benchmarks import BenchmarkRunner
        print("✓ All imports successful")
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False
    
    print("\nStep 6: Checking hardware...")
    try:
        from kernelmind.utils import get_device_info, detect_device
        device = detect_device()
        print(f"✓ Device detected: {device}")
        
        info = get_device_info()
        print(f"  Platform: {info.get('platform')}")
        print(f"  Processor: {info.get('processor')}")
    except Exception as e:
        print(f"⚠ Hardware check failed: {e}")
    
    print("\n" + "="*70)
    print("SETUP VERIFICATION COMPLETE")
    print("="*70)
    
    print("\nNext steps:")
    print("1. Run an example: python examples/simple_linear.py")
    print("2. Or launch interactive menu: python main.py")
    print("3. Read GETTING_STARTED.md for detailed documentation")
    
    print("\n✓ KERNELMIND is ready to use!")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
