"""
Run the web dashboard (Node.js).
This script just launches the Node.js server.
"""
import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).parent.parent
DASHBOARD_DIR = ROOT / "web" / "dashboard"


def main():
    # Check Node.js installed
    try:
        subprocess.run(["node", "--version"], check=True,
                       capture_output=True, text=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("[ERROR] Node.js not installed. Install from https://nodejs.org/")
        sys.exit(1)

    # Check node_modules
    nm = DASHBOARD_DIR / "node_modules"
    if not nm.exists():
        print("Installing dashboard dependencies...")
        subprocess.run(["npm", "install"], cwd=str(DASHBOARD_DIR), check=True)

    print("\n" + "=" * 50)
    print("  Starting dashboard on http://localhost:8080")
    print("  Press Ctrl+C to stop")
    print("=" * 50 + "\n")

    subprocess.run(["node", "src/server.js"], cwd=str(DASHBOARD_DIR))


if __name__ == "__main__":
    main()
