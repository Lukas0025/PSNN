dist: clean
	python3 setup.py sdist

install: dist
	pip3 install dist/*

upload: test
	twine upload dist/*

clean:
	rm -rf build
	rm -rf dist
	rm -rf PSNN.egg-info
	rm bin/PSNN

test: install
	@echo "\n\n----------------------"
	@echo "      start test        "
	@echo "----------------------\n\n"
	cd examples/ && for f in *.py; do python3 "$$f"; done
	@echo "\n\n----------------------"
	@echo "         pass        "
	@echo "----------------------\n\n"

