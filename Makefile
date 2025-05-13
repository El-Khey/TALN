.PHONY: setup clean install run

# Python environment variables
PYTHON=python3.12
VENV=.venv
VENV_BIN=$(VENV)/bin

# Default target
all: setup install

# Create Python virtual environment
setup:
	@echo "Creating virtual environment..."
	@$(PYTHON) -m venv $(VENV)

# Install required packages
install: setup
	@echo "Installing required packages..."
	@$(VENV_BIN)/pip install -r requirements.txt

# Run the notebook (opens it in VS Code)
run: install
	@echo "Opening notebook in VS Code..."
	@$(VENV_BIN)/python -m ipykernel install --user --name taln-env --display-name "Python (TALN)"
	@code text_preprocessing.ipynb

# Clean up
clean:
	@echo "Cleaning up..."
	@rm -rf $(VENV)
	@jupyter kernelspec remove -f taln-env 2>/dev/null || true

# Show help
help:
	@echo "Available commands:"
	@echo "  make setup    - Create Python virtual environment"
	@echo "  make install  - Install required packages"
	@echo "  make run      - Run the notebook in VS Code"
	@echo "  make clean    - Remove virtual environment and kernel"
	@echo "  make all      - Setup and install (default)"
	@echo "  make help     - Show this help message"