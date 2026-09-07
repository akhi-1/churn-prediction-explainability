.PHONY: install train explain impact test all

install:
	pip install -r requirements.txt

train:
	cd src && python train.py

explain:
	cd src && python explain.py

impact:
	cd src && python business_impact.py

test:
	python -m pytest tests/ -q

all: train explain impact
