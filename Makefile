# Makefile for ACE Bus Lane Violations Analysis

# Variables
PYTHON = python
PIP = pip
NOTEBOOK = jupyter notebook
LAB = jupyter lab

# Default target
all: install run

# Install dependencies
install:
	$(PIP) install -r requirements.txt

# Run the main analysis
run:
	$(PYTHON) src/ace_violations_analysis.py

# Start Jupyter Lab
lab:
	$(LAB)

# Start Jupyter Notebook
notebook:
	$(NOTEBOOK)

# Run data fetching only
fetch:
	$(PYTHON) -c "from src.ace_violations_analysis import ACEViolationsAnalyzer; analyzer = ACEViolationsAnalyzer(); analyzer.fetch_data()"

# Run processing only
process:
	$(PYTHON) -c "from src.ace_violations_analysis import ACEViolationsAnalyzer; analyzer = ACEViolationsAnalyzer(); analyzer.process_data()"

# Create visualizations only
visualize:
	$(PYTHON) -c "from src.ace_violations_analysis import ACEViolationsAnalyzer; analyzer = ACEViolationsAnalyzer(); analyzer.create_visualizations()"

# Open the HTML dashboard
dashboard:
	open index.html

# Clean up generated files
clean:
	rm -rf data/processed/*
	rm -rf data/raw/*
	rm -rf __pycache__/
	rm -rf .pytest_cache/
	find . -type f -name "*.pyc" -delete

# Help
help:
	@echo "Available targets:"
	@echo "  install    - Install Python dependencies"
	@echo "  run        - Run the complete analysis"
	@echo "  fetch      - Fetch data from APIs only"
	@echo "  process    - Process data only"
	@echo "  visualize  - Create visualizations only"
	@echo "  lab        - Start Jupyter Lab"
	@echo "  notebook   - Start Jupyter Notebook"
	@echo "  dashboard  - Open HTML dashboard"
	@echo "  clean      - Clean up generated files"
	@echo "  help       - Show this help message"
