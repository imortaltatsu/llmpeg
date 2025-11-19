# Publishing llmpeg to PyPI

## Prerequisites

1. **Create PyPI accounts:**
   - TestPyPI: https://test.pypi.org/account/register/
   - PyPI: https://pypi.org/account/register/

2. **Install build tools:**
   ```bash
   pip install build twine
   ```

## Step-by-Step Publishing Process

### 1. Test on TestPyPI First (Recommended)

```bash
# Build the package
python -m build

# Upload to TestPyPI
python -m twine upload --repository testpypi dist/*

# Test installation from TestPyPI
pip install --index-url https://test.pypi.org/simple/ llmpeg
```

### 2. Publish to Production PyPI

```bash
# Build the package (if not already built)
python -m build

# Upload to PyPI
python -m twine upload dist/*

# Verify installation
pip install llmpeg
```

## Using API Tokens (Recommended)

Instead of passwords, use API tokens:

1. **Create API token:**
   - Go to https://pypi.org/manage/account/token/
   - Create a new API token
   - Copy the token (starts with `pypi-`)

2. **Use token:**
   ```bash
   # When prompted for username, use: __token__
   # When prompted for password, use: pypi-<your-token>
   
   # Or set environment variables:
   export TWINE_USERNAME=__token__
   export TWINE_PASSWORD=pypi-<your-token>
   python -m twine upload dist/*
   ```

## Updating the Package

1. **Update version** in `pyproject.toml`:
   ```toml
   version = "0.1.1"  # Increment version number
   ```

2. **Build and upload:**
   ```bash
   python -m build
   python -m twine upload dist/*
   ```

## Troubleshooting

- **Package name already taken:** Choose a different name in `pyproject.toml`
- **Upload fails:** Make sure you're using the correct credentials
- **Build fails:** Check that all required files are present and `pyproject.toml` is valid

## Verification

After publishing, verify your package:
- Visit: https://pypi.org/project/llmpeg/
- Test installation: `pip install llmpeg`
- Test command: `llmpeg --help`

