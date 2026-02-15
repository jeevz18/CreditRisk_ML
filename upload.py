import subprocess
import os

print("="*70)
print("GITHUB UPLOAD - CLEAN VERSION")
print("="*70)
print()

# Get credentials
username = input("GitHub username (jeevz18): ").strip() or "jeevz18"
email = input("GitHub email: ").strip()
token = input("GitHub token (ghp_...): ").strip()
repo_name = input("Repository name (2025AA05523_CreditRisk_ML): ").strip() or "2025AA05523_CreditRisk_ML"

print("\nUploading...")

# Commands
commands = [
    f'git config user.name "{username}"',
    f'git config user.email "{email}"',
    'git init',
    'git add .',
    'git commit -m "Credit Risk ML - Complete Project"',
    'git branch -M main',
    f'git remote add origin https://github.com/{username}/{repo_name}.git',
    f'git push https://{username}:{token}@github.com/{username}/{repo_name}.git main --force'
]

for i, cmd in enumerate(commands, 1):
    print(f"\n[Step {i}/{len(commands)}] {cmd.split()[0]}...")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0 and i == len(commands):
        print(f"Error: {result.stderr}")
        if "404" in result.stderr or "not found" in result.stderr:
            print("\nRepository doesn't exist! Create it at:")
            print(f"https://github.com/new")
            print(f"Name: {repo_name}")
            print("Visibility: PUBLIC")
    elif i == len(commands):
        print(f"SUCCESS! View at: https://github.com/{username}/{repo_name}")

input("\nPress Enter to exit...")
