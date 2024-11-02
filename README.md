# Neuromorphic Time-Series Compression using Biological Memory Patterns with Muneeb's Algorithm

This project demonstrates data compression and analysis using Muneeb's Algorithm. The application visualizes the original and reconstructed data, calculates residuals, and provides an interactive quiz to test your understanding.

## Overview

- **Top Plot:** Shows how well the reconstructed data aligns with the original.
- **Bottom Plot:** Highlights differences between the original and reconstructed data.
- **Key Takeaway:** Smaller residuals mean better reconstruction.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/MuneebKhan11/Neuromorphic-Time-Series-Compression-using-Biological-Memory-Patterns.git
    cd repo-name
    ```

2. Create a virtual environment and activate it:
    ```sh
    python -m venv .venv
    source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Running the Application

To run the application, use the following command in the root of the project:
```sh
streamlit run app.py