#!/usr/bin/env python3
"""Quick syntax and import test"""
import sys
import traceback

try:
    print("Importing test v3...")
    # We'll just check if we can import and compile the module
    import py_compile
    py_compile.compile(r'c:\Users\Chris\Desktop\Gemini\test\test v3.py', doraise=True)
    print("✓ Syntax is valid!")
    
    print("\nAttempting to import all required modules...")
    import pandas as pd
    import numpy as np
    print("✓ pandas and numpy imported successfully")
    
    print("\n✓ All basic checks passed!")
    print("\nNote: Full backtest will run when you click 'Run Backtest' in the Dash app")
    
except Exception as e:
    print(f"✗ Error: {e}")
    traceback.print_exc()
    sys.exit(1)
