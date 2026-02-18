# Contributing to DBSIpy

First, thank you for your interest in contributing to **DBSIpy**! Contributions of all varieties and magnitudes are encouraged and valued.

If you like the project but don't have the time to contribute, that still helps: use DBSIpy (with appropriate citation in your work), suggest features you want to see, report bugs, or share the project with colleagues.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Questions](#questions)
- [How To Contribute](#how-to-contribute)
- [Reporting Bugs](#reporting-bugs)
- [Suggesting Enhancements](#suggesting-enhancements)
- [Improving The Documentation](#improving-the-documentation)
- [Style](#style)

## Code of Conduct

This project follows the Code of Conduct in [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md). The Code of Conduct applies to everyone involved in this project, regardless of contribution history. By participating, you are expected to uphold this code.

If you experience or observe unacceptable behavior, please report it to the community leaders responsible for enforcement at **devs.dbsipy@gmail.com**.

## Questions

Before asking a question, please:

- Read the project overview and usage notes in [README.rst](README.rst).
- Search existing [issues](/issues) to see whether your question has already been asked/answered.

If you find a similar issue but still need clarification, comment on that issue with your question and any additional context.

If no relevant issue exists, the most effective way to ask a question is:

- Open an [issue](/issues/new).
- Describe what you're trying to do and what happened instead.
- Include the DBSIpy version, Python version, OS, and (if relevant) PyTorch/CUDA info.

## How To Contribute

> ### Legal Notice
> By contributing to this project, you agree that you are the sole author of the contributed content, that you have the necessary rights to the content, and that the content you contribute may be distributed under the project license.

### Pull requests

DBSIpy uses GitHub pull requests for code and documentation changes.

- Fork the repository and create a feature branch from `main`.
- For development installs, follow [README.rst](README.rst) and install dev tooling with `requirements-dev.txt`.
- Make focused changes (small, reviewable PRs are easier to merge).
- Add or update tests when you change behavior.
- Run the test suite locally (`python -m pytest`).
- Open a pull request and clearly describe *what* changed and *why*.

## Reporting Bugs

### Before Submitting a Bug Report

A good bug report is detailed and reproducible. Before submitting, please:

- Make sure you are using the latest version of DBSIpy.
- Confirm the issue is not caused by user-side setup problems (e.g., environment mismatches, missing inputs, incorrect file paths).
- Search the [issue tracker](/issues) to see if the bug has already been reported.

Please collect:

- The full stack trace (traceback) and any relevant CLI output.
- OS name/version (Windows/Linux/macOS) and CPU architecture (x86_64/ARM).
- DBSIpy version (and how installed: from source, editable install, etc.).
- Python version.
- PyTorch version and whether CUDA is used (include CUDA version if applicable).
- If the issue depends on inputs/configuration:
  - the CLI command you ran, and
  - the configuration `.ini` (redact any private paths/data), and/or a minimal configuration that reproduces the issue.
- Whether the bug is reliably reproducible.

### How Do I Submit a Good Bug Report?

> Never report security-related issues, vulnerabilities, or bugs including sensitive information to the issue tracker, or elsewhere in public.
> Instead, please follow [SECURITY.md](SECURITY.md) and email **devs.dbsipy[at]gmail.com**.

For non-security bugs, DBSIpy uses GitHub issues:

- Open an [issue](/issues/new). Please do not add labels; a maintainer will triage.
- Explain expected behavior vs. observed behavior.
- Provide step-by-step reproduction instructions that someone else can follow.
- Include the information you collected above.

Once it's filed:

- A maintainer will triage and may ask for clarification.
- If we can reproduce it, we’ll label it for fixing and track progress in the issue.

## Suggesting Enhancements

Enhancement suggestions (new features or improvements) are tracked as [GitHub issues](/issues).

### Before Submitting an Enhancement

- Make sure you are using the latest version.
- Check whether the idea is already covered by an existing engine option, configuration setting, or workflow (see [README.rst](README.rst) and the example configs under `dbsipy/configs/`).
- Search existing [issues](/issues) to see whether the idea has already been suggested.
- Consider whether the enhancement fits DBSIpy’s scope (scientific diffusion MRI modeling and the DBSI/DBSI-IA/DTI workflows).

### How Do I Submit a Good Enhancement Suggestion?

- Use a clear, descriptive title.
- Describe the current behavior and the behavior you want.
- Explain why the change is valuable for DBSIpy users and how it should work.
- If possible, propose a minimal API/CLI/config change and any relevant references.

## Improving The Documentation

Documentation lives in the [docs/](docs/) folder and is built with Sphinx.

### Submitting a Documentation Improvement Suggestion

- Check whether the change is already addressed in existing docs (see `docs/source/`).
- Be specific about what page/section needs improvement and what should change.
- If you can, propose the exact wording or structure you’d like to see.

If you’re unsure whether it’s a bug or docs issue, open an [issue](/issues/new) and include “Documentation” in the title.

## Style

DBSIpy is written to be clear and easy to follow. We strongly prefer clean, human-readable code.

- Keep changes focused and consistent with existing patterns.
- Avoid obscure variable names and overly clever control flow.
- When changing behavior, add/adjust tests under `dbsipy/tests/` when feasible.
- Run the test suite (`python -m pytest`) before opening a PR.

Thank you again for your support of DBSIpy!

— The DBSIpy Team
