#!/bin/bash

# Set strict error handling
set -euo pipefail
IFS=$'\n\t'

# Initialize variables
LOGFILE="pipeline_$(date +%Y%m%d_%H%M%S).log"
REQUIRED_PYTHON_VERSION="3.8"
VENV_DIR="venv"
TEMP_DIR="tmp"
REQUIREMENTS_FILE="requirements.txt"

# Function to log messages
log() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] $1" | tee -a "$LOGFILE"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check Python version
check_python_version() {
    local python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    if (( $(echo "$python_version $REQUIRED_PYTHON_VERSION" | awk '{print ($1 >= $2)}') )); then
        log "Python version $python_version detected (meets minimum requirement of $REQUIRED_PYTHON_VERSION)"
        return 0
    else
        log "ERROR: Python version $python_version is less than required version $REQUIRED_PYTHON_VERSION"
        return 1
    fi
}

# Function to check GPU availability
check_gpu() {
    if command_exists nvidia-smi; then
        log "GPU detected: $(nvidia-smi --query-gpu=gpu_name --format=csv,noheader)"
        return 0
    else
        log "No GPU detected, running on CPU only"
        return 1
    fi
}

# Function to create requirements file
create_requirements() {
    cat > "$REQUIREMENTS_FILE" << EOF
pandas==1.5.3
numpy==1.23.0
matplotlib==3.7.1
seaborn==0.12.2
scipy==1.10.1
scikit-learn==1.2.0
EOF
    log "Created requirements file: $REQUIREMENTS_FILE"
}

# Function to verify dependencies
verify_dependencies() {
    local exit_code=0
    while IFS= read -r package; do
        python3 -c "import $(echo $package | cut -d'=' -f1)" 2>/dev/null || {
            log "WARNING: Package $package not properly installed"
            exit_code=1
        }
    done < "$REQUIREMENTS_FILE"
    return $exit_code
}

# Function to clean up temporary files
cleanup() {
    log "Performing cleanup..."
    if [ -d "$TEMP_DIR" ]; then
        rm -rf "$TEMP_DIR"/*
        log "Cleaned temporary directory"
    fi
    
    # Remove any matplotlib cache
    if [ -d ~/.cache/matplotlib ]; then
        rm -rf ~/.cache/matplotlib
        log "Cleaned matplotlib cache"
    fi
    
    # Keep only the last 5 log files
    ls -t pipeline_*.log | tail -n +6 | xargs -I {} rm -- {} 2>/dev/null || true
    log "Cleaned old log files"
}

# Function to handle script exit
handle_exit() {
    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        log "Script failed with exit code: $exit_code"
    fi
    
    # Deactivate virtual environment if it's active
    if [ -n "${VIRTUAL_ENV-}" ]; then
        deactivate
        log "Deactivated virtual environment"
    fi
    
    cleanup
    log "Script execution completed"
    exit $exit_code
}

# Register the exit handler
trap handle_exit EXIT

# Main execution starts here
main() {
    log "Starting ML pipeline execution"
    
    # Create temporary directory
    mkdir -p "$TEMP_DIR"
    
    # Check Python version
    check_python_version || {
        log "ERROR: Python version check failed"
        exit 1
    }
    
    # Check GPU availability
    check_gpu
    
    # Create requirements file
    create_requirements
    
    # Set up virtual environment
    if [ ! -d "$VENV_DIR" ]; then
        log "Creating virtual environment..."
        python3 -m venv "$VENV_DIR"
    else
        log "Virtual environment already exists"
    fi
    
    # Activate virtual environment
    log "Activating virtual environment..."
    source "$VENV_DIR/bin/activate"
    
    # Upgrade pip
    log "Upgrading pip..."
    pip install --upgrade pip > >(tee -a "$LOGFILE") 2>&1
    
    # Install dependencies
    log "Installing dependencies..."
    pip install -r "$REQUIREMENTS_FILE" > >(tee -a "$LOGFILE") 2>&1
    
    # Verify dependencies
    log "Verifying dependencies..."
    verify_dependencies || {
        log "ERROR: Dependency verification failed"
        exit 1
    }
    
    # Check if the ML pipeline script exists
    if [ ! -f "end_to_end_ml_pipeline.py" ]; then
        log "ERROR: end_to_end_ml_pipeline.py not found"
        exit 1
    }
    
    # Run the ML pipeline
    log "Executing ML pipeline..."
    python end_to_end_ml_pipeline.py 2>&1 | tee -a "$LOGFILE"
    
    # Check if pipeline execution was successful
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        log "ML pipeline executed successfully"
    else
        log "ERROR: ML pipeline execution failed"
        exit 1
    fi
}

# Execute main function
main "$@"
