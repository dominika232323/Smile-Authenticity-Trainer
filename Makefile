#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = Smile-Authenticity-Trainer
PYTHON_VERSION = 3.11
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Set up Python environment
.PHONY: create_environment
create_environment:
	@rm -rf venv
	$(PYTHON_INTERPRETER)$(PYTHON_VERSION) -m venv venv
	@echo ">>> New python interpreter environment created. Activate it using 'source venv/bin/activate'"


## Install Python dependencies
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt


## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Delete preprocessed data
.PHONY: remove_preprocessed_data
remove_preprocessed_data:
	rm -r data/original_data
	rm -r data/preprocessed_UvA-NEMO_SMILE_DATABASE


## Delete logs
.PHONY: remove_logs
remove_logs:
	rm logs/app.log
	rm logs/errors.log


## Delete runs
.PHONY: remove_runs
remove_runs:
	rm -r runs/


## Sort imports using isort
.PHONY: isort
isort:
	isort data_preprocessing modeling


## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check


## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format


## Check type hints with mypy
.PHONY: mypy
mypy:
	mypy data_preprocessing modeling api


#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Run data preprocessing
.PHONY: data_preprocessing
data_preprocessing:
	$(PYTHON_INTERPRETER) -m data_preprocessing.main


## Train lips model on features
.PHONY: train_lips_features
train_lips_features:
	$(PYTHON_INTERPRETER) -m modeling.lips_features


## Train eyes model on features
.PHONY: train_eyes_features
train_eyes_features:
	$(PYTHON_INTERPRETER) -m modeling.eyes_features


## Train cheeks model on features
.PHONY: train_cheeks_features
train_cheeks_features:
	$(PYTHON_INTERPRETER) -m modeling.cheeks_features


## Train model on all features
.PHONY: train_all_features
train_all_features:
	$(PYTHON_INTERPRETER) -m modeling.all_features


## Train lips model on landmarks
.PHONY: train_lips_landmarks
train_lips_landmarks:
	$(PYTHON_INTERPRETER) -m modeling.lips_landmarks


## Train eyes model on landmarks
.PHONY: train_eyes_landmarks
train_eyes_landmarks:
	$(PYTHON_INTERPRETER) -m modeling.eyes_landmarks


## Train cheeks model on landmarks
.PHONY: train_cheeks_landmarks
train_cheeks_landmarks:
	$(PYTHON_INTERPRETER) -m modeling.cheeks_landmarks


## Run api
.PHONY: run_api
run_api:
	$(PYTHON_INTERPRETER) -m api.main


## Test api
.PHONY: test_api
test_api:
	curl -X POST http://127.0.0.1:5000/process-video -F "video=@data/UvA-NEMO_SMILE_DATABASE/videos/001_deliberate_smile_2.mp4"


## Run tests
.PHONY: test
test:
	$(PYTHON_INTERPRETER) -m pytest tests


## Run tests with coverage
.PHONY: coverage
coverage:
	$(PYTHON_INTERPRETER) -m pytest --cov=data_preprocessing --cov=modeling --cov=api --cov-report=term-missing tests


## Run tensorboard for lips features runs
.PHONY: tensorboard_lips_features
tensorboard_lips_features:
	tensorboard --logdir=runs/lips_runs/


## Run tensorboard for eyes features runs
.PHONY: tensorboard_eyes_features
tensorboard_eyes_features:
	tensorboard --logdir=runs/eyes_runs/


## Run tensorboard for cheeks features runs
.PHONY: tensorboard_cheeks_features
tensorboard_cheeks_features:
	tensorboard --logdir=runs/cheek_runs/


## Run tensorboard for all features runs
.PHONY: tensorboard_all_features
tensorboard_all_features:
	tensorboard --logdir=runs/all_features_runs/


## Run tensorboard for lips landmarks runs
.PHONY: tensorboard_lips_landmarks
tensorboard_lips_landmarks:
	tensorboard --logdir=runs/lips_landmarks_runs/


## Run tensorboard for eyes landmarks runs
.PHONY: tensorboard_eyes_landmarks
tensorboard_eyes_landmarks:
	tensorboard --logdir=runs/eyes_landmarks_runs/


## Run tensorboard for cheeks landmarks runs
.PHONY: tensorboard_cheeks_landmarks
tensorboard_cheeks_landmarks:
	tensorboard --logdir=runs/cheek_landmarks_runs/



#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
