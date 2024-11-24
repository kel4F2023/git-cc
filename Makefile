.PHONY: install clean

ENV_NAME := $(shell which pip)

install:
	@echo "Installing git-cc tool with pip at $(ENV_NAME) "
	@pip install .
	@echo "Done"

clean:
	@echo "Uninstalling git-cc tool "
	@pip uninstall git-cc -y
	@echo "Cleaning up build files"
	@rm -rf build src/git_cc.egg-info
	@echo "Done"

train:
	@echo "Training the model"
	@python base/train.py
	@echo "Done"