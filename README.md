#Named Entity Recognition (NER)

This project implements a Named Entity Recognition (NER) system using fine-tuned transformer models. It combines the OntoNotes dataset with custom SpaceX-related data to train a model capable of identifying entities like persons, organizations, locations, and more.

## Overview

The project uses DistilBERT as the base model and fine-tunes it on a combined dataset consisting of:
- **OntoNotes 5.0**: A large-scale corpus for general domain NER
- **Custom Dataset**: SpaceX and space exploration-related text with entity annotations

## Features

- Data preparation and preprocessing pipeline
- Model training with configurable epochs
- Entity prediction on custom text
- Performance evaluation with detailed classification reports
- GPU/CPU support with interactive device selection

## Requirements

### Standard Installation
```bash
pip install -r requirements.txt
```

### GPU Installation (for CUDA support)
```bash
pip install -r requirements_gpu.txt
```

### Dependencies
- transformers==4.57.3
- datasets==2.19.1
- evaluate==0.4.6
- seqeval==1.2.2
- accelerate==1.12.0
- scikit-learn==1.7.2
- numpy==2.2.6
- torch==2.5.1

## Project Structure

```
.
├── prepare_data.py          # Data preparation and preprocessing
├── train.py                 # Model training script
├── pipeline.py              # Inference pipeline for predictions
├── generate_report.py       # Evaluation and classification report
├── run_project.sh           # End-to-end execution script
├── dataset2_formatted.json  # Custom SpaceX dataset
├── train.json              # Training data (generated)
├── validation.json         # Validation data (generated)
├── project_test_sentences.txt  # Test sentences
├── requirements.txt        # Python dependencies
└── requirements_gpu.txt    # GPU-specific dependencies
```

## Usage

### Quick Start

Run the entire pipeline with the provided script:

```bash
bash run_project.sh
```

This will execute all three steps sequentially.

### Step-by-Step Execution

#### 1. Prepare Data

Prepare the training and validation datasets:

```bash
python prepare_data.py
```

This script:
- Downloads the OntoNotes 5.0 dataset
- Splits it into 85% training and 15% validation
- Merges custom dataset (if available)
- Converts label IDs to string format
- Saves `train.json` and `validation.json`

#### 2. Train Model

Train the NER model:

```bash
python train.py --model_save_path "./my_model" --num_train_epoch 3
```

Parameters:
- `--model_save_path`: Directory to save the trained model (required)
- `--num_train_epoch`: Number of training epochs (default: 3)

Training configuration:
- Base model: `distilbert-base-uncased`
- Learning rate: 2e-5
- Batch size: 16
- Evaluation strategy: Every epoch
- Metric: F1 score

#### 3. Run Predictions

Use the trained model to predict entities in new text:

```bash
python pipeline.py --model_load_path "./my_model" --input_file "project_test_sentences.txt" --output_file "final_results.json"
```

Parameters:
- `--model_load_path`: Path to the saved model folder (required)
- `--input_file`: Text file with sentences to analyze (required)
- `--output_file`: Output JSON file for results (default: "final_results.json")

The pipeline supports interactive GPU/CPU selection when GPU is available.

#### 4. Generate Evaluation Report

Evaluate model performance on the validation set:

```bash
python generate_report.py
```

This script:
- Loads the trained model from `./my_model`
- Evaluates on `validation.json`
- Displays overall metrics (accuracy, precision, recall, F1)
- Shows per-entity classification report

## Input/Output Format

### Input (Text File)
One sentence per line:
```
Elon Musk founded SpaceX in California.
The Falcon 9 rocket launched from Cape Canaveral.
```

### Output (JSON)
```json
[
  {
    "input": "Elon Musk founded SpaceX in California.",
    "entities": [
      {
        "entity_group": "PERSON",
        "score": 0.99,
        "word": "Elon Musk",
        "start": 0,
        "end": 9
      },
      {
        "entity_group": "ORG",
        "score": 0.98,
        "word": "SpaceX",
        "start": 18,
        "end": 24
      },
      {
        "entity_group": "GPE",
        "score": 0.97,
        "word": "California",
        "start": 28,
        "end": 38
      }
    ]
  }
]
```

## Entity Types

The model recognizes the following entity types (OntoNotes standard):
- **PERSON**: People, including fictional characters
- **ORG**: Organizations, companies, agencies, institutions
- **GPE**: Geopolitical entities (countries, cities, states)
- **LOC**: Non-GPE locations, mountain ranges, bodies of water
- **DATE**: Absolute or relative dates or periods
- **TIME**: Times smaller than a day
- **MONEY**: Monetary values
- **PERCENT**: Percentages
- **PRODUCT**: Objects, vehicles, foods, etc.
- **EVENT**: Named events
- **WORK_OF_ART**: Titles of books, songs, etc.
- **LAW**: Named documents made into laws
- **LANGUAGE**: Named languages
- **QUANTITY**: Measurements
- **ORDINAL**: Ordinal numbers
- **CARDINAL**: Cardinal numbers
- **FAC**: Buildings, airports, highways, bridges

## Notes

- The model uses the BIO tagging scheme (B-prefix for beginning, I-prefix for inside)
- Token aggregation strategy is set to "simple" to merge sub-word tokens
- Training uses the best model based on F1 score
- GPU acceleration is automatically detected and can be interactively selected

## License

This project is intended for educational purposes.

## Acknowledgments

- OntoNotes 5.0 dataset from `tner/ontonotes5`
- Hugging Face Transformers library
- DistilBERT model from Hugging Face
