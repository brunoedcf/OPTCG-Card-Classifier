# OPTCG-Card-Classifier

## Prerequisites

Before starting, you will need to have Python installed on your machine. This project was tested with Python 3.10.11. Additionally, you will need Git to clone the repository.

## Installation

Clone the repository to your local machine using the following command in your terminal:

```bash
git clone https://github.com/brunoedcf/OPTCG-Card-Classifier.git
cd OPTCG-Card-Classifier
python -m venv venv
./venv/Scripts/Activate.ps1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install ultralytics