.PHONY: setup clean install run vocab help check-python

# Python environment variables
PYTHON_VERSION := 3.10
VENV=.venv
VENV_BIN=$(VENV)/bin

# Default target
all: setup install

# Create Python virtual environment
setup:
	@echo "Checking Python $(PYTHON_VERSION) availability..."
	@if ! command -v python$(PYTHON_VERSION) &> /dev/null; then \
		echo "Python $(PYTHON_VERSION) is not installed. Installing..."; \
		sudo apt update && sudo apt install -y python$(PYTHON_VERSION) python$(PYTHON_VERSION)-venv; \
	fi
	@echo "Creating virtual environment with Python $(PYTHON_VERSION)..."
	@python$(PYTHON_VERSION) -m venv $(VENV) || (echo "Failed to create virtual environment. Please ensure python$(PYTHON_VERSION)-venv is installed."; exit 1)
	@echo "Upgrading pip..."
	@$(VENV_BIN)/pip install --upgrade pip

# Install required packages
install: setup
	@echo "Installing TensorFlow and OpenNMT-tf..."
	@$(VENV_BIN)/pip install "tensorflow>=2.10,<2.14" opennmt-tf
	@$(VENV_BIN)/pip install pandas scikit-learn

install-requirements:
	@echo "Installing additional requirements from requirements.txt..."
	@$(VENV_BIN)/pip install -r requirements.txt

# Run the notebook (opens it in VS Code)
run: install
	@echo "Setting up Jupyter kernel..."
	@$(VENV_BIN)/python -m ipykernel install --user --name onmt-env --display-name "Python (OpenNMT)"
	@echo "Opening notebook..."
	@code text_preprocessing.ipynb

# Generate vocabularies
vocab: install
	@echo "Preparing training and test data from CSV and generating vocabularies..."
	@$(VENV_BIN)/python utils/data_splitter.py --input data/cleaned/fr_en_processed_data.csv --output-dir data/cleaned/ --test-size 0.2 --vocab-size 5000

# Clean up
clean:
	@echo "Cleaning up..."
	@rm -rf $(VENV)
	@jupyter kernelspec remove -f onmt-env 2>/dev/null || true

# Show help
help:
	@echo "Available commands:"
	@echo "  make setup    - Create Python virtual environment"
	@echo "  make install  - Install required packages"
	@echo "  make run      - Run the notebook in VS Code"
	@echo "  make vocab    - Generate vocabularies for OpenNMT"
	@echo "  make clean    - Remove virtual environment and kernel"
	@echo "  make all      - Setup and install (default)"
	@echo "  make help     - Show this help message"