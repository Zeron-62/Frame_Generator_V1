# Publishing FrameForge on GitHub

This repository is prepared so runtime files and model weights stay out of Git.

## Create the repository

On GitHub, create a new repository named for example:

```text
FrameForge-In-Between
```

Do **not** upload model weights or generated output files.

## First upload from Windows

Open Command Prompt or PowerShell in the project directory:

```bat
git init
git add .
git status
git commit -m "Initial FrameForge In-Between release"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/FrameForge-In-Between.git
git push -u origin main
```

Replace `YOUR_USERNAME` and the repository name with your GitHub repository details.

## Before pushing

Run:

```bat
python -m py_compile app.py
git status
```

Confirm that these are **not** staged:

```text
.venv/
output/
logs/
temp_inputs/
*.safetensors
*.ckpt
*.pt
*.pth
*.bin
```

The `.gitignore` file already excludes them.

## Releases

For a tagged release, for example:

```bat
git tag -a v4.0.0 -m "FrameForge In-Between v4.0.0"
git push origin v4.0.0
```

GitHub Releases can then be used for source archives or future packaged application builds.
