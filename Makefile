build: setup.py
	python3 setup.py sdist

install:
	pip3 install dist/*

upload:
	twine upload dist/*

clean:
	rm -rf build
	rm -rf dist
	rm -rf PYSNN.egg-info
