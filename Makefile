.PHONY: all init test docs install clean

all: test

init:
	pip install -r requirements.txt

test:
	nosetests tests

docs:
	sphinx-build -b html docs build

install:
	python setup.py install

clean:
	rm -rf *.pyc
	rm -rf build
