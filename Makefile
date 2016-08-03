.PHONY: all init test install

all: test

init:
	pip install -r requirements.txt

test:
	nosetests tests

install:
	python setup.py install
