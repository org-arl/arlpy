.PHONY: all init test

all: test

init:
	pip install -r requirements.txt

test:
	nosetests tests
