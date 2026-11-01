.PHONY: install app test demo
install:
	pip install -r requirements.txt
app:
	streamlit run app.py
test:
	python -m pytest -q
demo:
	python -m contextswap examples/synthetic_input.png --bg "#2E86DE" --engine grabcut --keep-shadows -o out/demo.png
