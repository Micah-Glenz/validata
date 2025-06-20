# PyPI Publishing Setup Guide

## 🔐 Step 1: Create PyPI Accounts

### Test PyPI (for testing):
1. Go to: https://test.pypi.org/account/register/
2. Create account and verify email
3. Enable 2FA (recommended)

### Production PyPI:
1. Go to: https://pypi.org/account/register/
2. Create account and verify email  
3. Enable 2FA (recommended)

## 🔑 Step 2: Create API Tokens

### Test PyPI Token:
1. Go to: https://test.pypi.org/manage/account/token/
2. Click "Add API token"
3. Name: "validata-package-upload"
4. Scope: "Entire account" (or specific to validata project if it exists)
5. Copy the token (starts with `pypi-`)

### Production PyPI Token:
1. Go to: https://pypi.org/manage/account/token/
2. Click "Add API token"
3. Name: "validata-package-upload"
4. Scope: "Entire account" (or specific to validata project if it exists)
5. Copy the token (starts with `pypi-`)

## 🚀 Step 3: Upload Package

### Method 1: Environment Variables
```bash
# For Test PyPI
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=pypi-YOUR_TEST_TOKEN_HERE
export TWINE_REPOSITORY=testpypi
twine upload dist/*

# For Production PyPI
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=pypi-YOUR_PRODUCTION_TOKEN_HERE
twine upload dist/*
```

### Method 2: Interactive Upload
```bash
# Activate virtual environment
source venv/bin/activate

# Test upload first
twine upload --repository testpypi dist/*
# When prompted:
# Username: __token__
# Password: pypi-YOUR_TEST_TOKEN_HERE

# Production upload
twine upload dist/*
# When prompted:
# Username: __token__
# Password: pypi-YOUR_PRODUCTION_TOKEN_HERE
```

### Method 3: Configuration File
Create `~/.pypirc`:
```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
repository = https://upload.pypi.org/legacy/
username = __token__
password = pypi-YOUR_PRODUCTION_TOKEN_HERE

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-YOUR_TEST_TOKEN_HERE
```

## ✅ Step 4: Verify Upload

### Test PyPI:
- View package: https://test.pypi.org/project/validata/
- Test install: `pip install --index-url https://test.pypi.org/simple/ validata`

### Production PyPI:
- View package: https://pypi.org/project/validata/
- Install: `pip install validata`

## 🔄 Future Releases

### Version Bump Process:
1. Update version in `setup.py` (e.g., `0.1.1`)
2. Create git tag: `git tag v0.1.1`
3. Push tag: `git push origin v0.1.1`
4. Rebuild package: `python -m build`
5. Upload: `twine upload dist/*`

### Package Contents:
- Source: `validata-{version}.tar.gz`
- Wheel: `validata-{version}-py3-none-any.whl`

## 🛡️ Security Notes:
- Never commit API tokens to git
- Use environment variables or secure credential storage
- Enable 2FA on PyPI accounts
- Use project-scoped tokens when possible
- Regularly rotate API tokens

## 📊 Package Info:
- **Name**: validata
- **Current Version**: 0.1.0
- **License**: MIT
- **Python**: 3.8+
- **Repository**: https://github.com/Micah-Glenz/validata