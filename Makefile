# some makefile magic commands
.DEFAULT_GOAL := HELP
.PHONY: HELP

# include ENV variables in .env file if it exists
ifeq (,$(wildcard .env))
$(error ERROR: .env file not found - please create an .env file with S3 credentials)
else
include .env
endif

env:  ## create your environment with uv
	@echo Creating environment with uv
	@source $$(pwd)/.env
	@uv sync

pypi-publish: env  ## build and upload to pypi (set up .env with PYPI_TOKEN)
	@echo Removing \"dist\" directory if present
	@rm -rf dist *.egg-info
	@echo 
	uv build
	@echo
	@echo publish --token '$$(PYPI_TOKEN)'
	-@uv publish --token $(PYPI_TOKEN)
	@echo
	@echo Removing \"dist\" directory
	@rm -rf dist *.egg-info

HELP: # show this help
	@echo
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = "(:|## )"}; {printf "\033[36m%-15s\033[0m %s\n", $$2, $$4}'
	@echo
