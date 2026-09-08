.PHONY: synthetic train serve eval

synthetic:
	python -m src.router.generate_synthetic

train:
	python -m src.router.train

serve:
	uvicorn src.gateway.main:app --reload --port 8000

eval:
	python -m src.eval.shadow_harness
