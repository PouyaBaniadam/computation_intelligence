import sys
import os
import time
import runpy

from utils import Colors, resource_path


def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_banner():
    print(f"{Colors.HEADER}{Colors.BOLD}" + "=" * 50)
    print("     Computational Intelligence - HW2 Launcher")
    print("=" * 50 + f"{Colors.ENDC}")

def main():
    while True:
        clear_screen()
        print_banner()
        print(f"\n{Colors.CYAN}Please select the implementation method:{Colors.ENDC}")
        print(f"{Colors.GREEN}[1]{Colors.ENDC} Pure Python Implementation (Without Toolbox)")
        print(f"{Colors.GREEN}[2]{Colors.ENDC} NumPy Implementation (Vectorized/Fast)")
        print(f"{Colors.FAIL}[0]{Colors.ENDC} Exit")

        choice = input(f"\n{Colors.BOLD}Enter your choice (0-2): {Colors.ENDC}")

        script_to_run = ""
        method_name = ""

        if choice == '1':
            script_to_run = "pure_nn.py"
            method_name = "Pure Python"
        elif choice == '2':
            script_to_run = "numpy_nn.py"
            method_name = "NumPy"
        elif choice == '0':
            print(f"\n{Colors.BLUE}Goodbye!{Colors.ENDC}")
            sys.exit()
        else:
            print(f"\n{Colors.FAIL}Invalid choice!{Colors.ENDC}")
            time.sleep(1)
            continue

        full_script_path = resource_path(script_to_run)

        if not os.path.exists(full_script_path):
             print(f"\n{Colors.FAIL}Error: File '{script_to_run}' not found!{Colors.ENDC}")
             time.sleep(2)
             continue

        print(f"\n{Colors.BLUE}{Colors.BOLD}>>> Running {method_name} implementation...{Colors.ENDC}\n")

        start_time = time.time()

        try:
            runpy.run_path(full_script_path, run_name="__main__")
        except Exception as e:
            print(f"\n{Colors.FAIL}An error occurred:{Colors.ENDC}")
            print(e)

        end_time = time.time()
        duration = end_time - start_time

        print(f"\n{Colors.HEADER}{Colors.BOLD}" + "-" * 50)
        print(f" Execution Finished for: {method_name}")
        print(f" Total Time: {Colors.GREEN}{duration:.4f} seconds{Colors.ENDC}")
        print("-" * 50 + f"{Colors.ENDC}")

        input(f"\n{Colors.BOLD}Press Enter to return to menu...{Colors.ENDC}")

if __name__ == "__main__":
    main()
