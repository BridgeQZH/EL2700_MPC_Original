import subprocess
import sys

def install(package):
    print(subprocess.check_call([sys.executable, "-m", "pip", "install", package]))

if __name__ == "__main__":
    install("matplotlib")
    install("casadi")
    install("numpy")
    install("control")

    try:
        import matplotlib
    except :
        print("ERROR: Matplotlib package not installed. Please install it manually for your environment")
        exit()
    
    try:
        import casadi
    except :
        print("ERROR: CasADi package not installed. Please install it manually for your environment")
        exit()

    try:
        import numpy
    except :
        print("ERROR: Numpy package not installed. Please install it manually for your environment")
        exit()

    try:
        import control
    except :
        print("ERROR: Control package not installed. Please install it manually for your environment")
        exit()

