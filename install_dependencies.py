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
PYTORCH_PACKAGES = {"torch", "torchvision", "torchaudio"}
MEGA_ASR_REQUIREMENTS = os.path.join("Mega-ASR", "requirements.txt")


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
    command = [sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio"]
    if wheel_tag:
        command.extend(["--index-url", f"https://download.pytorch.org/whl/{wheel_tag}"])
    else:
        command.extend(["--index-url", "https://download.pytorch.org/whl/cpu"])

    print("Installing PyTorch:")
    print(" ".join(command))
    if not dry_run:
        run(command)


def requirement_name(requirement):
    match = re.match(r"\s*([A-Za-z0-9_.-]+)", requirement)
    return match.group(1).lower().replace("_", "-") if match else ""


def read_requirements(path, excluded_packages=None):
    excluded_packages = {name.lower().replace("_", "-") for name in (excluded_packages or set())}
    requirements = []
    skipped = []

    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.split("#", 1)[0].strip()
            if not line:
                continue

            name = requirement_name(line)
            if name in excluded_packages:
                skipped.append(line)
                continue

            requirements.append(line)

    return requirements, skipped


def install_requirements_file(path, label, dry_run=False, excluded_packages=None):
    if excluded_packages:
        requirements, skipped = read_requirements(path, excluded_packages)
        if skipped:
            print(f"Skipping {label} PyTorch pins because PyTorch was installed above:")
            for item in skipped:
                print(f"  {item}")
        if not requirements:
            print(f"No installable {label} requirements found in {path}.")
            return
        command = [sys.executable, "-m", "pip", "install", *requirements]
    else:
        command = [sys.executable, "-m", "pip", "install", "-r", path]

    print(f"Installing {label} requirements:")
    print(" ".join(command))
    if not dry_run:
        run(command)


def install_requirements(dry_run=False):
    install_requirements_file("requirements.txt", "Qontex", dry_run=dry_run)


def install_mega_asr_requirements(dry_run=False):
    if not os.path.exists(MEGA_ASR_REQUIREMENTS):
        print(f"Mega-ASR requirements not found at {MEGA_ASR_REQUIREMENTS}; skipping.")
        return

    install_requirements_file(
        MEGA_ASR_REQUIREMENTS,
        "Mega-ASR",
        dry_run=dry_run,
        excluded_packages=PYTORCH_PACKAGES,
    )


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
    parser.add_argument("--skip-mega-asr", action="store_true", help="Do not install Mega-ASR requirements.")
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
    if args.skip_mega_asr:
        print("Skipping Mega-ASR requirements by request.")
    else:
        install_mega_asr_requirements(dry_run=args.dry_run)
    verify_torch(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
