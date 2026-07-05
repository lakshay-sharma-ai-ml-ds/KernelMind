#!/usr/bin/env python3

import sys
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from kernelmind.utils import get_logger, print_hardware_info
from kernelmind.config import config

logger = get_logger(__name__)

def print_banner():
    banner = """
    
╔════════════════════════════════════════════════════════════════╗
║                                                                ║
║     KERNELMIND: Agentic ML Compiler & GPU Kernel Optimizer     ║
║                                                                ║
║       Analyze , Optimize , Generate , Benchmark , Deploy       ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝

"""
    print(banner)

def print_menu():
    menu = """
KERNELMIND Examples & Tools
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Simple Linear Model Optimization
   - Basic example with MLPs
   - Graph parsing and optimization
   - Kernel generation and benchmarking
   
2. ResNet-18 Optimization
   - Real-world CNN model
   - Multi-layer fusion
   - Comprehensive benchmarking
   
3. Transformer Block Fusion
   - Attention mechanism optimization
   - FFN kernel generation
   - Decision engine demonstration
   
4. Hardware Information
   - Display system capabilities
   - Device detection
   - Memory profiling
   
5. Run All Examples
   - Execute all optimization examples
   - Generate complete reports
   
0. Exit

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"""
    print(menu)

def main():
    print_banner()
    
    print("Checking configuration...")
    try:
        config.validate()
        print("✓ Configuration valid\n")
    except ValueError as e:
        print(f"✗ Configuration error: {e}")
        print("  Make sure ANTHROPIC_API_KEY is set in environment or .env file")
        return
    
    while True:
        print_menu()
        choice = input("Select option (0-5): ").strip()
        
        if choice == "0":
            print("Exiting KERNELMIND. Goodbye!")
            sys.exit(0)
        
        elif choice == "1":
            print("\n" + "="*70)
            print("Running Simple Linear Model Optimization Example")
            print("="*70 + "\n")
            try:
                from kernelmind.examples import run_simple_linear_example
                run_simple_linear_example()
            except Exception as e:
                print(f"Error: {e}")
                logger.error(f"Example failed: {e}", exc_info=True)
        
        elif choice == "2":
            print("\n" + "="*70)
            print("Running ResNet-18 Optimization Example")
            print("="*70 + "\n")
            try:
                from kernelmind.examples import run_resnet_example
                run_resnet_example()
            except Exception as e:
                print(f"Error: {e}")
                logger.error(f"Example failed: {e}", exc_info=True)
        
        elif choice == "3":
            print("\n" + "="*70)
            print("Running Transformer Layer Fusion Example")
            print("="*70 + "\n")
            try:
                from kernelmind.examples.transformer_fusion import run_transformer_example
                run_transformer_example()
            except Exception as e:
                print(f"Error: {e}")
                logger.error(f"Example failed: {e}", exc_info=True)
        
        elif choice == "4":
            print("\nSystem Information:")
            print("-" * 70)
            print_hardware_info()
        
        elif choice == "5":
            print("\nRunning all examples...\n")
            try:
                print("\n1. Simple Linear Model")
                print("-" * 70)
                from kernelmind.examples import run_simple_linear_example
                run_simple_linear_example()
                
                print("\n2. ResNet-18")
                print("-" * 70)
                from kernelmind.examples import run_resnet_example
                run_resnet_example()
                
                print("\n3. Transformer Block")
                print("-" * 70)
                from kernelmind.examples.transformer_fusion import run_transformer_example
                run_transformer_example()
                
                print("\nAll examples completed successfully!")
            
            except Exception as e:
                print(f"Error running examples: {e}")
                logger.error(f"Examples failed: {e}", exc_info=True)
        
        else:
            print("Invalid option. Please try again.\n")
        
        input("\nPress Enter to continue...")
        print("\n" * 2)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
