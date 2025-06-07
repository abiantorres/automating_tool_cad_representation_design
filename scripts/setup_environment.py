#!/usr/bin/env python3
"""
Simple and robust script to setup the ATCRD development environment
"""

import subprocess
import sys
import os
from pathlib import Path

# Configuration
ENV_NAME = "atcrd_env"
ENV_FILE = "config/environments/atcrd_env.yml"
CADLIB_PATH = "src/cadlib"

def run_cmd(cmd, description="Running command"):
    """Run a command with simple output"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(
            cmd, shell=True, check=True, 
            capture_output=True, text=True
        )
        if result.stdout.strip():
            print(f"   {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {e}")
        if e.stderr:
            print(f"   Error: {e.stderr.strip()}")
        return False

def check_conda():
    """Check if conda is available"""
    if run_cmd("conda --version", "Checking conda"):
        return True
    print("❌ Conda not found. Please install Anaconda or Miniconda.")
    return False

def get_current_env():
    """Get the current conda environment"""
    try:
        result = subprocess.run(
            ["conda", "info", "--envs"], 
            capture_output=True, text=True, check=True
        )
        for line in result.stdout.split('\n'):
            if '*' in line:
                env_name = line.split()[0]
                return env_name if env_name != 'base' else None
    except:
        pass
    return None

def environment_exists():
    """Check if environment already exists"""
    result = subprocess.run(
        ["conda", "env", "list"], 
        capture_output=True, text=True
    )
    return ENV_NAME in result.stdout

def remove_existing_env():
    """Remove existing environment with proper deactivation"""
    current_env = get_current_env()
    
    if current_env == ENV_NAME:
        print(f"⚠️  You are currently in the '{ENV_NAME}' environment")
        print("   Please run this command first:")
        print(f"   conda deactivate && python scripts/setup_environment.py")
        return False
    
    return run_cmd(
        f"conda env remove -n {ENV_NAME} -y", 
        f"Removing existing '{ENV_NAME}' environment"
    )

def create_environment():
    """Create conda environment"""
    # Try using the YAML file first
    if Path(ENV_FILE).exists():
        print(f"📦 Creating environment from {ENV_FILE}...")
        if run_cmd(f"conda env create -f {ENV_FILE}", "Creating environment from YAML"):
            return True
        print("⚠️  YAML creation failed, trying manual creation...")
    
    # Fallback to manual creation with proper channels
    print("📦 Creating environment manually...")
    
    # Create basic environment
    cmd = (
        f"conda create -n {ENV_NAME} "
        f"python=3.9 numpy matplotlib pip pytest black flake8 "
        f"-c conda-forge -y"
    )
    
    if not run_cmd(cmd, "Creating basic environment"):
        return False
    
    # Install pythonocc-core from conda-forge
    if not run_cmd(
        f"conda install -n {ENV_NAME} pythonocc-core -c conda-forge -c dlr-sc -y",
        "Installing pythonocc-core"
    ):
        print("⚠️  pythonocc-core installation failed, trying alternative...")
        # Try without dlr-sc channel
        run_cmd(
            f"conda install -n {ENV_NAME} pythonocc-core -c conda-forge -y",
            "Installing pythonocc-core (alternative)"
        )
    
    # Install pip dependencies
    pip_deps = ["trimesh"]
    for dep in pip_deps:
        run_cmd(
            f"conda run -n {ENV_NAME} pip install {dep}", 
            f"Installing {dep}"
        )
    
    return True

def install_cadlib():
    """Install cadlib package in development mode"""
    cadlib_path = Path(CADLIB_PATH)
    
    if not cadlib_path.exists():
        print(f"⚠️  CADLib not found at {cadlib_path}")
        print("   Skipping cadlib installation")
        return False
    
    return run_cmd(
        f"conda run -n {ENV_NAME} pip install -e {cadlib_path}", 
        "Installing cadlib in development mode"
    )

def verify_installation():
    """Verify the installation"""
    print("✅ Verifying installation...")
    
    # Test basic imports
    test_cmd = (
        f"conda run -n {ENV_NAME} python -c "
        f'"import numpy, matplotlib; print(\\"✓ Basic packages work\\")"'
    )
    
    if not run_cmd(test_cmd, "Testing basic packages"):
        return False
    
    # Test pythonocc-core
    occ_test = (
        f"conda run -n {ENV_NAME} python -c "
        f'"try: import OCC; print(\\"✓ pythonocc-core works\\"); except: print(\\"⚠️  pythonocc-core not available\\")"'
    )
    run_cmd(occ_test, "Testing pythonocc-core")
    
    # Test trimesh
    trimesh_test = (
        f"conda run -n {ENV_NAME} python -c "
        f'"try: import trimesh; print(\\"✓ trimesh works\\"); except: print(\\"⚠️  trimesh not available\\")"'
    )
    run_cmd(trimesh_test, "Testing trimesh")
    
    # Test cadlib if available
    if Path(CADLIB_PATH).exists():
        cadlib_test = (
            f"conda run -n {ENV_NAME} python -c "
            f'"try: import cadlib; print(f\\"✓ CADLib v{{cadlib.__version__}} ready\\"); except Exception as e: print(f\\"⚠️  CADLib issue: {{e}}\\")"'
        )
        run_cmd(cadlib_test, "Testing cadlib")
    
    return True

def main():
    """Main setup function"""
    print("🚀 Setting up ATCRD development environment")
    print(f"   Environment name: {ENV_NAME}")
    print(f"   Config file: {ENV_FILE}")
    print("=" * 50)
    
    # Check prerequisites
    if not check_conda():
        sys.exit(1)
    
    # Handle existing environment
    if environment_exists():
        current_env = get_current_env()
        print(f"⚠️  Environment '{ENV_NAME}' already exists")
        
        if current_env == ENV_NAME:
            print("❌ You are currently in the environment you want to recreate!")
            print("\n🔧 Please run these commands:")
            print("   conda deactivate")
            print("   python scripts/setup_environment.py")
            sys.exit(1)
        
        response = input("Remove and recreate? (y/N): ").strip().lower()
        if response == 'y':
            if not remove_existing_env():
                sys.exit(1)
        else:
            print("❌ Aborting setup")
            sys.exit(1)
    
    # Create environment
    if not create_environment():
        print("❌ Failed to create environment")
        sys.exit(1)
    
    # Install cadlib
    install_cadlib()
    
    # Verify everything works
    if verify_installation():
        print("\n🎉 Setup completed successfully!")
    else:
        print("\n⚠️  Setup completed with some issues")
    
    # Usage instructions
    print("\n" + "=" * 50)
    print("📋 Next steps:")
    print(f"   conda activate {ENV_NAME}")
    print("   python -c \"import cadlib; print('Ready!')\"")
    print("\n💡 To deactivate:")
    print("   conda deactivate")

if __name__ == "__main__":
    main()