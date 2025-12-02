import sys
import os
import time
import runpy


class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

def clear_screen():
    if os.name != 'nt' and 'TERM' not in os.environ:
        os.environ['TERM'] = 'xterm'

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
        print(f"{Colors.GREEN}[2]{Colors.ENDC} NumPy Implementation (With Toolbox/NumPy)")
        print(f"{Colors.FAIL}[0]{Colors.ENDC} Exit")

        choice = input(f"\n{Colors.BOLD}Enter your choice (0-2): {Colors.ENDC}")

        script_to_run = ""
        method_name = ""

        if choice == '1':
            script_to_run = resource_path("without_toolbox.py")
            method_name = "Pure Python"
        elif choice == '2':
            script_to_run = resource_path("with_toolbox.py")
            method_name = "NumPy / Vectorized"
        elif choice == '0':
            print(f"\n{Colors.BLUE}Goodbye!{Colors.ENDC}")
            sys.exit()
        else:
            print(f"\n{Colors.FAIL}Invalid choice! Please try again.{Colors.ENDC}")
            time.sleep(1)
            continue

        print(f"\n{Colors.BLUE}{Colors.BOLD}>>> Running {method_name} implementation...{Colors.ENDC}\n")

        start_time = time.time()

        try:
            runpy.run_path(script_to_run, run_name="__main__")
        except Exception as e:
            print(f"\n{Colors.FAIL}An error occurred during execution:{Colors.ENDC}")
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