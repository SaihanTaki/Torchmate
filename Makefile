BG = \033[46m # background color
FG = \033[30m # foreground color
RESET = \033[0m # back to default color
TORCH_VERSION = 2.2.1
VENV_NAME = .venv


.PHONY: test clean

$(VENV_NAME):
	@echo -e "$(FG)$(BG)Creating Virtual Environment.......$(RESET)"
	python -m venv $(VENV_NAME)



# Installation Rules

install_torch_cpu:
	@echo -e "$(FG)$(BG)Installing PyTorch $(TORCH_VERSION)+cpu.......$(RESET)"
	$(VENV_NAME)/bin/python -m pip install --upgrade pip			
	$(VENV_NAME)/bin/pip install torch==$(TORCH_VERSION)+cpu -f https://download.pytorch.org/whl/torch_stable.html

install_dev: $(VENV_NAME) install_torch_cpu
	@echo -e "$(FG)$(BG)Installing Dependencies.......$(RESET)"
	$(VENV_NAME)/bin/pip install -r requirements.txt
	@echo -e "$(FG)$(BG)Installing Optional Dependencies (for linting, testing etc).......$(RESET)"
	$(VENV_NAME)/bin/pip install -r requirements_dev.txt
	$(VENV_NAME)/bin/pre-commit install

install_doc_req: $(VENV_NAME)
	@echo -e "$(FG)$(BG)Installing documentation dependencies.......$(RESET)"
	$(VENV_NAME)/bin/python -m pip install --upgrade pip
	$(VENV_NAME)/bin/pip install -r docs/requirements.txt

install_pypi_req: $(VENV_NAME)
	@echo -e "$(FG)$(BG)Installing dependencies for building and uploading to pypi.......$(RESET)"
	$(VENV_NAME)/bin/python -m pip install --upgrade pip
	$(VENV_NAME)/bin/pip install setuptools twine wheel

install_all: install_dev install_doc_req install_pypi_req
	@echo -e "$(FG)$(BG)All the dependencies for the package, testing, styling, pypi and documentation are installed!$(RESET)"



# Styling and Testing Rules

test: $(VENV_NAME)
	@echo -e "$(FG)$(BG)Testing the Package.......$(RESET)"
	$(VENV_NAME)/bin/python -m pytest --cov=torchmate --cov-report xml:coverage.xml --cov-report term --cov-config=.coveragerc 

black: $(VENV_NAME)
	@echo -e "$(FG)$(BG)Running Black.......$(RESET)"
	$(VENV_NAME)/bin/black torchmate/ --config=pyproject.toml --check

flake8: $(VENV_NAME)
	@echo -e "$(FG)$(BG)Running Flake8.......$(RESET)"
	$(VENV_NAME)/bin/flake8 torchmate/ --config=.flake8

isort: .venv
	@echo -e "$(FG)$(BG)Running Isort.......$(RESET)"
	.venv/bin/isort torchmate/ --settings=pyproject.toml

clean:
	@echo -e "$(FG)$(BG)Cleaning Caches.......$(RESET)"
	rm -rf .pytest_cache __pycache__ torchmate.egg-info .coverage
	rm -rf $(shell find torchmate/  -name "__pycache__" -type d)
	rm -rf $(shell find tests/  -name ".pytest_cache" -type d)
	rm -rf $(shell find tests/  -name ".coverage" -type f)
	rm -rf build dist
	rm -f coverage.xml

all: clean black flake8 isort test
