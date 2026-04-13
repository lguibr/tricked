import argparse
import subprocess
import sys
import os
import signal

def run_subprocess(cmd, cwd=None, env=None):
    print(f"Running: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, cwd=cwd, env=env)
    return proc

def main():
    parser = argparse.ArgumentParser(description="Tricked AI DevX Controller")
    parser.add_argument("layer", choices=["dev", "backend", "frontend"], help="Target layer or dev environment")
    parser.add_argument("action", nargs="?", choices=["format", "lint", "build", "test"], help="Action to perform")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.abspath(__file__))
    frontend_dir = os.path.join(project_root, "frontend")
    backend_dir = os.path.join(project_root, "backend")
    engine_dir = os.path.join(backend_dir, "engine")
    venv_python = os.path.join(project_root, "venv", "bin", "python")
    pytest = os.path.join(project_root, "venv", "bin", "pytest")

    env = os.environ.copy()
    env["PYTHONPATH"] = project_root
    env["LIBTORCH_BYPASS_VERSION_CHECK"] = "1"
    env["LIBTORCH_USE_PYTORCH"] = "1"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    try:
        import torch
        torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
        if "LD_LIBRARY_PATH" in env:
            env["LD_LIBRARY_PATH"] = f"{torch_lib}:{env['LD_LIBRARY_PATH']}"
        else:
            env["LD_LIBRARY_PATH"] = torch_lib
    except ImportError:
        pass

    if args.layer == "dev":
        print("Starting Tricked AI Development Environment...")
        
        # Auto-kill any leftover processes hanging on our development ports (8000 for Uvicorn, 1420 for Vite)
        try:
            import psutil
            for port in [8000, 1420]:
                for conn in psutil.net_connections(kind='inet'):
                    if conn.laddr.port == port and conn.status == 'LISTEN' and conn.pid:
                        try:
                            proc = psutil.Process(conn.pid)
                            print(f"Auto-killing {proc.name()} (PID {conn.pid}) to free up port {port}...")
                            for child in proc.children(recursive=True):
                                child.kill()
                            proc.kill()
                            proc.wait(timeout=2)
                        except Exception:
                            pass
        except ImportError:
            print("psutil not installed; unable to automatically clear ports.")
        except Exception as e:
            print(f"Failed to auto-kill ports: {e}")

        backend_proc = run_subprocess([venv_python, "-m", "uvicorn", "backend.server:app", "--reload"], cwd=project_root)
        frontend_proc = run_subprocess(["npm", "run", "dev"], cwd=frontend_dir)
        
        def signal_handler(sig, frame):
            print("\nShutting down dev environment...")
            backend_proc.terminate()
            frontend_proc.terminate()
            sys.exit(0)
            
        signal.signal(signal.SIGINT, signal_handler)
        backend_proc.wait()
        frontend_proc.wait()
        return

    if not args.action:
        print("Please specify an action: format, lint, build, or test")
        sys.exit(1)

    if args.layer == "backend":
        if args.action == "format":
            print("Formatting Rust Engine...")
            sys.exit(run_subprocess(["cargo", "fmt"], cwd=engine_dir, env=env).wait())
        elif args.action == "lint":
            print("Linting Rust Engine...")
            sys.exit(run_subprocess(["cargo", "clippy", "--", "-D", "warnings"], cwd=engine_dir, env=env).wait())
        elif args.action == "build":
            print("Building Rust Engine bindings in release mode...")
            ret = run_subprocess(["cargo", "build", "--release"], cwd=engine_dir, env=env).wait()
            if ret != 0: sys.exit(ret)
            so_src = os.path.join(project_root, "target", "release", "libtricked_engine.so")
            so_dst = os.path.join(backend_dir, "tricked_engine.so")
            if os.path.exists(so_src):
                import shutil
                shutil.copy2(so_src, so_dst)
                print(f"Installed native Python bindings: {so_dst}")
        elif args.action == "test":
            print("Running Rust Internal Unit Tests...")
            ret = run_subprocess(["cargo", "test"], cwd=engine_dir, env=env).wait()
            if ret != 0: sys.exit(ret)
            print("\nRunning Python ML Integration Tests...")
            sys.exit(run_subprocess([pytest, os.path.join(backend_dir, "tests")], cwd=project_root, env=env).wait())

    elif args.layer == "frontend":
        if args.action == "format":
            sys.exit(run_subprocess(["npm", "run", "format"], cwd=frontend_dir).wait())
        elif args.action == "lint":
            sys.exit(run_subprocess(["npm", "run", "lint"], cwd=frontend_dir).wait())
        elif args.action == "build":
            sys.exit(run_subprocess(["npm", "run", "build"], cwd=frontend_dir).wait())
        elif args.action == "test":
            sys.exit(run_subprocess(["npm", "run", "test"], cwd=frontend_dir).wait())

if __name__ == "__main__":
    main()
