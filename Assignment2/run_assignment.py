"""
CVI620 Assignment 2 - Setup and Execution Helper
Run this script first to set up the environment
"""

import os
import sys
import subprocess

def print_header(text):
    print("\n" + "="*60)
    print(text)
    print("="*60 + "\n")

def check_python_version():
    """Check if Python version is compatible"""
    print_header("Checking Python Version")
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required!")
        return False
    
    print("✓ Python version is compatible")
    return True

def install_requirements():
    """Install required packages"""
    print_header("Installing Requirements")
    
    requirements_file = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    
    if not os.path.exists(requirements_file):
        print("❌ requirements.txt not found!")
        return False
    
    print("Installing packages from requirements.txt...")
    print("This may take several minutes...\n")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_file])
        print("\n✓ All packages installed successfully!")
        return True
    except subprocess.CalledProcessError:
        print("\n❌ Error installing packages!")
        return False

def check_datasets():
    """Check if datasets are present"""
    print_header("Checking Datasets")
    
    # Check Q1 dataset
    q1_train_cat = os.path.join("Q1", "train", "Cat")
    q1_train_dog = os.path.join("Q1", "train", "Dog")
    q1_test_cat = os.path.join("Q1", "test", "Cat")
    q1_test_dog = os.path.join("Q1", "test", "Dog")
    
    q1_ok = all(os.path.exists(p) for p in [q1_train_cat, q1_train_dog, q1_test_cat, q1_test_dog])
    
    if q1_ok:
        cat_count = len(os.listdir(q1_train_cat))
        dog_count = len(os.listdir(q1_train_dog))
        print(f"✓ Q1 Dataset Found:")
        print(f"  - Training: {cat_count} cats, {dog_count} dogs")
        print(f"  - Test: {len(os.listdir(q1_test_cat))} cats, {len(os.listdir(q1_test_dog))} dogs")
    else:
        print("❌ Q1 Dataset (Cat/Dog) not found!")
    
    # Check Q2 dataset
    q2_train = os.path.join("Q2", "mnist_train.csv")
    q2_test = os.path.join("Q2", "mnist_test.csv")
    
    q2_ok = os.path.exists(q2_train) and os.path.exists(q2_test)
    
    if q2_ok:
        print(f"✓ Q2 Dataset Found:")
        print(f"  - mnist_train.csv")
        print(f"  - mnist_test.csv")
    else:
        print("❌ Q2 Dataset (MNIST) not found!")
    
    return q1_ok and q2_ok

def show_menu():
    """Show main menu"""
    print_header("CVI620 Assignment 2 - Main Menu")
    
    print("Q1: Cat vs Dog Classification")
    print("  1. Train Q1 models (all 4 models)")
    print("  2. Run Q1 inference")
    print()
    print("Q2: MNIST Digit Classification")
    print("  3. Train Q2 models (all 6 classifiers)")
    print("  4. Run Q2 inference")
    print()
    print("Other Options")
    print("  5. Install/Update requirements")
    print("  6. Check system setup")
    print("  0. Exit")
    print()

def run_script(script_path, working_dir):
    """Run a Python script in a specific directory"""
    original_dir = os.getcwd()
    try:
        os.chdir(working_dir)
        print(f"\n➜ Running: {script_path}")
        print(f"➜ Working directory: {working_dir}\n")
        subprocess.run([sys.executable, script_path])
        print("\n✓ Script completed!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
    finally:
        os.chdir(original_dir)
    
    input("\nPress Enter to continue...")

def main():
    """Main function"""
    print_header("CVI620 Assignment 2 - Setup & Execution Helper")
    
    # Initial checks
    if not check_python_version():
        print("\nPlease upgrade Python and try again.")
        return
    
    print("\nChecking system setup...")
    check_datasets()
    
    while True:
        show_menu()
        choice = input("Enter your choice: ").strip()
        
        if choice == '0':
            print("\nGoodbye!")
            break
        
        elif choice == '1':
            run_script("train_catdog.py", "Q1")
        
        elif choice == '2':
            run_script("inference_catdog.py", "Q1")
        
        elif choice == '3':
            run_script("train_mnist.py", "Q2")
        
        elif choice == '4':
            run_script("inference_mnist.py", "Q2")
        
        elif choice == '5':
            install_requirements()
            input("\nPress Enter to continue...")
        
        elif choice == '6':
            check_python_version()
            check_datasets()
            
            print("\nChecking installed packages...")
            try:
                import tensorflow
                print(f"✓ TensorFlow: {tensorflow.__version__}")
            except ImportError:
                print("❌ TensorFlow not installed")
            
            try:
                import sklearn
                print(f"✓ scikit-learn: {sklearn.__version__}")
            except ImportError:
                print("❌ scikit-learn not installed")
            
            try:
                import pandas
                print(f"✓ pandas: {pandas.__version__}")
            except ImportError:
                print("❌ pandas not installed")
            
            try:
                import matplotlib
                print(f"✓ matplotlib: {matplotlib.__version__}")
            except ImportError:
                print("❌ matplotlib not installed")
            
            input("\nPress Enter to continue...")
        
        else:
            print("\n❌ Invalid choice! Please try again.")
            input("\nPress Enter to continue...")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        input("\nPress Enter to exit...")
