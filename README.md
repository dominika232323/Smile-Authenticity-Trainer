# Smile-Authenticity-Trainer

Smile Authenticity Trainer is a mobile application supported by a machine-learning–based backend whose goal is to help users train a more authentic smile.

The project combines video data preprocessing, feature extraction, machine learning model training, and a mobile application that presents analysis results to the user.

## Project goal

The goal of this project is to design an intelligent system that supports users in developing a more authentic smile.

## Features

- Video data preprocessing
- Feature extraction
- Machine learning model training
- Mobile application

## Project Structure

```
.
├── api/                         # Backend for the mobile application (API)
├── data/                        # Dataset (UvA-NEMO Smile Database)
├── data_preprocessing/          # Data preprocessing and feature extraction
├── logs/                        # Application log files
├── modeling/                    # Machine learning models and training scripts
├── models/                      # Trained models and configurations
├── runs/                        # Training experiment results (TensorBoard logs)
├── smile_authenticity_trainer/  # Mobile application source code
├── tests/                       # Unit and integration tests
├── config.py                    # Global project configuration
├── logging_config.py            # Logging configuration
├── Makefile                     # Task automation
├── pyproject.toml               # Project and tool configuration
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation
├── .gitignore                   # Git ignore rules
└── .env                         # Environment variables
```

## Dataset

The project uses the [UvA-NEMO Smile Database](https://www.uva-nemo.org/), which contains facial video recordings presenting both genuine and posed smiles. The dataset is used for training and evaluating the smile authenticity classification models.

The dataset is not distributed with this repository and must be obtained separately and placed in the `data/` directory.

## Installation and setup

Python 3.11 is required.

### Create a virtual environment

```bash
make create_environment
source venv/bin/activate
```

### Install dependencies

```bash
make requirements
```

## Data Preprocessing

To preprocess the dataset (feature extraction and dataset preparation), run:

```bash
make data_preprocessing
```

## Model training

Models can be trained separately for different facial regions and data representations. Run the following commands to train the models on different configurations of hyperparameters:

#### Feature-based models

```bash
make train_lips_features
make train_eyes_features
make train_cheeks_features
make train_all_features
```

#### Landmark-based models

```bash
make train_lips_landmarks
make train_eyes_landmarks
make train_cheeks_landmarks
```

## Training Visualization (TensorBoard)

To view the training progress, run:

```bash
make tensorboard_lips_features
make tensorboard_eyes_features
make tensorboard_cheeks_features
make tensorboard_all_features
make tensorboard_lips_landmarks
make tensorboard_eyes_landmarks
make tensorboard_cheeks_landmarks
```

## Testing

Run unit and integration tests:

```bash
make test
```

Run tests with a coverage report:

```bash
make coverage
```

## Running the API

To start the backend API for the mobile application:

```bash
make run_api
```

To run API `.env` file is required. It should contain:

```
GEMINI_API_KEY=YOUR_KEY
```

## Development tools

### Code linting

```bash
make lint
```

### Code formatting

```bash
make format
```

### Static type checking

```bash
make mypy
```
