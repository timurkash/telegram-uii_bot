
run:
	@uvicorn main:app --reload

freeze:
	@pip freeze > requirements.txt