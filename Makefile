all: install

install:
	python setup.py install

develop:
	python setup.py develop

develop-uninstall:
	python setup.py develop --uninstall
