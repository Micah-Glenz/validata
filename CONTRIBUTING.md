# Contributing to Validata

Thank you for your interest in contributing to Validata! This document provides guidelines for contributing to the project.

## Code of Conduct

This project adheres to a code of conduct. By participating, you are expected to uphold this code.

## How to Contribute

### Reporting Bugs

1. Check if the bug has already been reported in [Issues](https://github.com/Micah-Glenz/validata/issues)
2. Use the bug report template when creating a new issue
3. Include a minimal code example that reproduces the bug
4. Provide environment details (OS, Python version, etc.)

### Suggesting Features

1. Check existing [Issues](https://github.com/Micah-Glenz/validata/issues) for similar requests
2. Use the feature request template
3. Clearly describe the use case and expected behavior
4. Consider providing a proposed API design

### Contributing Code

#### Setup Development Environment

```bash
# Fork the repository and clone your fork
git clone https://github.com/YOUR_USERNAME/validata.git
cd validata

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Run tests to ensure everything works
python test_validation_schema_comprehensive.py
```

#### Development Workflow

1. **Create a branch** for your feature/fix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following these guidelines:
   - Write clear, self-documenting code
   - Add docstrings to new functions/classes
   - Follow existing code style
   - Add tests for new functionality

3. **Test your changes**:
   ```bash
   python test_validation_schema_comprehensive.py
   ```

4. **Commit your changes**:
   ```bash
   git add .
   git commit -m "feat: add new validation feature"
   ```

5. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Create a Pull Request** using the provided template

#### Code Style Guidelines

- Follow PEP 8 Python style guide
- Use descriptive variable and function names
- Write docstrings for all public functions and classes
- Keep functions focused and small
- Add type hints where appropriate

#### Testing Guidelines

- All new features must include tests
- Tests should cover both success and failure cases
- Use descriptive test names
- Ensure tests pass before submitting PR

## Review Process

1. All pull requests require review before merging
2. Maintainers will review code for:
   - Functionality and correctness
   - Code quality and style
   - Test coverage
   - Documentation completeness

3. Feedback will be provided via PR comments
4. Address feedback and update your PR as needed

## Release Process

Releases are managed by maintainers and follow semantic versioning:
- **Patch** (0.1.X): Bug fixes
- **Minor** (0.X.0): New features, backward compatible
- **Major** (X.0.0): Breaking changes

## Questions?

If you have questions about contributing, please:
1. Check existing documentation
2. Search [Issues](https://github.com/Micah-Glenz/validata/issues)
3. Create a new issue with the question label

Thank you for contributing to Validata!