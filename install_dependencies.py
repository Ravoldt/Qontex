import argparse
import os
import re
import subprocess
import sys


PYTORCH_CUDA_WHEELS = (
    ((12, 8), "cu128"),
    ((12, 6), "cu126"),
    ((11, 8), "cu118"),
)

PREFERRED_CUDA_MAJOR = 12


def run(command, check=True, capture_output=False):
    return subprocess.run(
        command,
        check=check,
        text=True,
        stdout=subprocess.PIPE if capture_output else None,
        stderr=subprocess.STDOUT if capture_output else None,
    )


def parse_cuda_version(text):
    patterns = (
        r"CUDA Version:\s*(\d+)\.(\d+)",
        r"release\s+(\d+)\.(\d+)",
        r"\bv?(\d+)\.(\d+)\b",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return int(match.group(1)), int(match.group(2))
    return None


def add_cuda_version(versions, version, source):
    if version:
        versions.setdefault(version, set()).add(source)


def detect_cuda_versions():
    versions = {}

    command_checks = (
        (["nvidia-smi"], "nvidia-smi driver runtime"),
        (["nvcc", "--version"], "nvcc on PATH"),
    )
    for command, source in command_checks:
        try:
            result = run(command, check=False, capture_output=True)
        except FileNotFoundError:
            continue

        add_cuda_version(versions, parse_cuda_version(result.stdout or ""), source)

    for env_name, env_value in os.environ.items():
        if not env_value or "CUDA" not in env_name.upper():
            continue
        add_cuda_version(versions, parse_cuda_version(env_value), f"{env_name} environment variable")

    cuda_roots = (
        os.environ.get("CUDA_PATH"),
        os.environ.get("CUDA_HOME"),
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA",
        "/usr/local",
    )
    for root in cuda_roots:
        if not root or not os.path.isdir(root):
            continue

        try:
            entries = os.listdir(root)
        except OSError:
            continue

        for entry in entries:
            version = parse_cuda_version(entry)
            if version:
                add_cuda_version(versions, version, os.path.join(root, entry))

    return versions


def select_cuda_wheel(cuda_versions):
    if not cuda_versions:
        return None

    sorted_versions = sorted(cuda_versions.keys(), reverse=True)
    preferred_versions = [version for version in sorted_versions if version[0] == PREFERRED_CUDA_MAJOR]
    if preferred_versions:
        return "cu126", preferred_versions[0]
    return None


def format_version(version):
    return f"{version[0]}.{version[1]}"


def print_detected_cuda_versions(cuda_versions):
    if not cuda_versions:
        print("No CUDA installations detected.")
        return

    print("Detected CUDA versions:")
    for version in sorted(cuda_versions.keys(), reverse=True):
        sources = ", ".join(sorted(cuda_versions[version]))
        print(f"  CUDA {format_version(version)}: {sources}")


def install_pytorch(wheel_tag, dry_run=False):
    command = [sys.executable, "-m", "pip", "install", "torch"]
    if wheel_tag:
        command.extend(["--index-url", f"https://download.pytorch.org/whl/{wheel_tag}"])
    else:
        command.extend(["--index-url", "https://download.pytorch.org/whl/cpu"])

    print("Installing PyTorch:")
    print(" ".join(command))
    if not dry_run:
        run(command)


def install_requirements(dry_run=False):
    command = [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"]
    print("Installing remaining requirements:")
    print(" ".join(command))
    if not dry_run:
        run(command)


def verify_torch(dry_run=False):
    command = [
        sys.executable,
        "-c",
        "import torch; print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available()); print('torch_cuda', torch.version.cuda)",
    ]
    print("Verifying PyTorch:")
    print(" ".join(command))
    if not dry_run:
        run(command)


def main():
    parser = argparse.ArgumentParser(description="Install Qontex dependencies with automatic PyTorch CUDA selection.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU-only PyTorch.")
    parser.add_argument("--cuda", choices=[tag for _, tag in PYTORCH_CUDA_WHEELS], help="Force a specific PyTorch CUDA wheel.")
    parser.add_argument("--dry-run", action="store_true", help="Print install commands without running them.")
    args = parser.parse_args()

    cuda_versions = {} if args.cpu else detect_cuda_versions()
    selected = None if args.cpu or args.cuda else select_cuda_wheel(cuda_versions)
    wheel_tag = args.cuda or (selected[0] if selected else None)
    selected_cuda_version = selected[1] if selected else None

    if args.cpu:
        print("CPU-only install requested.")
    elif args.cuda:
        print(f"Using requested PyTorch CUDA wheel: {args.cuda}")
        print_detected_cuda_versions(cuda_versions)
    elif wheel_tag:
        print_detected_cuda_versions(cuda_versions)
        print(f"Using PyTorch wheel {wheel_tag} based on CUDA {format_version(selected_cuda_version)}.")
    else:
        print_detected_cuda_versions(cuda_versions)
        print("CUDA 12 was not detected. Using CPU-only PyTorch because faster-whisper pip wheels require CUDA 12.")

    install_pytorch(wheel_tag, dry_run=args.dry_run)
    install_requirements(dry_run=args.dry_run)
    verify_torch(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
