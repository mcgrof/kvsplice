# Makefile for KV-Compress experiments
#
# Main targets:
#   make all       - Run all experiments and generate plots
#   make test      - Run experiments only (no plots)
#   make plots     - Generate plots from existing results
#   make clean     - Remove generated files
#   make help      - Show this help message

PYTHON := python3
VENV := .venv
VENV_PYTHON := $(VENV)/bin/python3
VENV_PIP := $(VENV)/bin/pip

# Experiment parameters
EPOCHS ?= 12
K_VALUES ?= 8,12,16,24,32,48,64

# Result files
SPLINE_RESULTS := results_spline_pca.csv
FLOW_RESULTS := results_affineflow_pca.csv

# Plot files
PLOTS := compression_comparison.png \
         improvement_delta.png \
         memory_reduction.png \
         architecture_diagram.png

.PHONY: all test plots clean help venv check-venv

# Default target: run everything
all: check-venv test plots
	@echo ""
	@echo "✓ All experiments complete!"
	@echo ""
	@echo "Results:"
	@echo "  - $(SPLINE_RESULTS)"
	@echo "  - $(FLOW_RESULTS)"
	@echo ""
	@echo "Plots:"
	@for plot in $(PLOTS); do echo "  - $$plot"; done
	@echo ""
	@echo "To view plots, open any PNG file in an image viewer."

# Run experiments only
test: check-venv test-spline test-flow
	@echo "✓ Experiments complete. Run 'make plots' to generate visualizations."

# Run Spline→PCA experiment
test-spline: check-venv
	@echo "=========================================="
	@echo "Running Spline→PCA experiment..."
	@echo "  Epochs: $(EPOCHS)"
	@echo "  k values: $(K_VALUES)"
	@echo "=========================================="
	@$(VENV_PYTHON) splinepca.py --epochs $(EPOCHS) --k-values $(K_VALUES)
	@echo ""
	@echo "✓ Spline→PCA results saved to $(SPLINE_RESULTS)"
	@echo ""

# Run Affine Flow→PCA experiment
test-flow: check-venv
	@echo "=========================================="
	@echo "Running Affine Flow→PCA experiment..."
	@echo "  Epochs: $(EPOCHS)"
	@echo "=========================================="
	@$(VENV_PYTHON) affineflow_pca_experiment.py --epochs $(EPOCHS)
	@echo ""
	@echo "✓ Flow→PCA results saved to $(FLOW_RESULTS)"
	@echo ""

# Generate all plots
plots: check-venv $(SPLINE_RESULTS) $(FLOW_RESULTS)
	@echo "=========================================="
	@echo "Generating visualization plots..."
	@echo "=========================================="
	@$(VENV_PYTHON) plot_results.py
	@echo ""
	@echo "✓ Plots generated successfully!"
	@echo ""

# Quick test (fewer epochs for faster validation)
quick: check-venv
	@echo "Running quick validation (2 epochs)..."
	@$(MAKE) test EPOCHS=2
	@$(MAKE) plots
	@echo "✓ Quick test complete!"

# Create virtual environment
venv:
	@if [ ! -d "$(VENV)" ]; then \
		echo "Creating virtual environment..."; \
		$(PYTHON) -m venv $(VENV); \
		echo "Installing dependencies..."; \
		$(VENV_PIP) install --upgrade pip; \
		$(VENV_PIP) install torch numpy pandas matplotlib; \
		echo "✓ Virtual environment ready!"; \
		echo ""; \
		echo "To activate manually:"; \
		echo "  source $(VENV)/bin/activate"; \
	else \
		echo "Virtual environment already exists."; \
	fi

# Check if virtual environment exists
check-venv:
	@if [ ! -d "$(VENV)" ]; then \
		echo "Virtual environment not found. Creating..."; \
		$(MAKE) venv; \
	fi

# Clean generated files
clean:
	@echo "Cleaning generated files..."
	@rm -f $(SPLINE_RESULTS) $(FLOW_RESULTS)
	@rm -f $(PLOTS)
	@echo "✓ Cleaned results and plots."

# Deep clean (including venv)
distclean: clean
	@echo "Removing virtual environment..."
	@rm -rf $(VENV)
	@rm -rf __pycache__
	@echo "✓ Deep clean complete."

# Show help
help:
	@echo "KV-Compress Makefile"
	@echo "===================="
	@echo ""
	@echo "Targets:"
	@echo "  make all        - Run all experiments and generate plots (default)"
	@echo "  make test       - Run experiments only (Spline→PCA + Flow→PCA)"
	@echo "  make plots      - Generate visualization plots from results"
	@echo "  make quick      - Fast validation (2 epochs instead of $(EPOCHS))"
	@echo "  make venv       - Create Python virtual environment"
	@echo "  make clean      - Remove generated results and plots"
	@echo "  make distclean  - Remove everything including venv"
	@echo "  make help       - Show this message"
	@echo ""
	@echo "Options:"
	@echo "  EPOCHS=N        - Number of training epochs (default: $(EPOCHS))"
	@echo "  K_VALUES=...    - Comma-separated k values (default: $(K_VALUES))"
	@echo ""
	@echo "Examples:"
	@echo "  make all                    # Run everything with defaults"
	@echo "  make test EPOCHS=8          # Run experiments with 8 epochs"
	@echo "  make quick                  # Fast test (2 epochs)"
	@echo "  make plots                  # Regenerate plots from existing results"
	@echo ""
	@echo "First time setup:"
	@echo "  1. make venv                # Create virtual environment"
	@echo "  2. make all                 # Run experiments and generate plots"
	@echo ""
	@echo "Results:"
	@echo "  - $(SPLINE_RESULTS)         # Spline→PCA numerical results"
	@echo "  - $(FLOW_RESULTS)           # Flow→PCA numerical results"
	@echo ""
	@echo "Plots:"
	@for plot in $(PLOTS); do echo "  - $$plot"; done

# Ensure result files exist before plotting
$(SPLINE_RESULTS):
	@echo "Error: $(SPLINE_RESULTS) not found. Run 'make test-spline' first."
	@exit 1

$(FLOW_RESULTS):
	@echo "Error: $(FLOW_RESULTS) not found. Run 'make test-flow' first."
	@exit 1
