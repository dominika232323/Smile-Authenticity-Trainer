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
	mypy data_preprocessing modeling modeling_2


#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Run data preprocessing
.PHONY: data_preprocessing
data_preprocessing:
	$(PYTHON_INTERPRETER) -m data_preprocessing.main


## Train lips model
.PHONY: train_lips
train_lips:
	$(PYTHON_INTERPRETER) -m modeling.lips


## Train eyes model
.PHONY: train_eyes
train_eyes:
	$(PYTHON_INTERPRETER) -m modeling.eyes


## Train cheeks model
.PHONY: train_cheeks
train_cheeks:
	$(PYTHON_INTERPRETER) -m modeling.cheeks


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
	$(PYTHON_INTERPRETER) -m pytest --cov=data_preprocessing --cov-report=term-missing tests


## Run tensorboard for lips runs
.PHONY: tensorboard_lips
tensorboard_lips:
	tensorboard --logdir=runs/lips_runs/


## Run tensorboard for eyes runs
.PHONY: tensorboard_eyes
tensorboard_eyes:
	tensorboard --logdir=runs/eyes_runs/


## Run tensorboard for cheeks runs
.PHONY: tensorboard_cheeks
tensorboard_cheeks:
	tensorboard --logdir=runs/cheek_runs/



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
