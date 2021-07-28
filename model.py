import json
import math
from data_parsing import MAX_AGE, MAX_HOURS

with open("weights.json", "r") as file:
	WEIGHTS = json.load(file)

def predict(age: int, has_college_degree: bool, is_married: bool, is_male: bool,
	weekly_work_hours: int) -> float:
	'''
	Predicts whether a person makes at least 50k dollars a year or not.
	:returns: The probability that this person makes over 50k dollars.
	'''

	# We have to standardize my input data the same way we did when training the model.
	age = age / MAX_AGE
	weekly_work_hours = weekly_work_hours / MAX_HOURS

	s = (age*WEIGHTS[0] + has_college_degree*WEIGHTS[1] + is_married*WEIGHTS[2] +
			is_male*WEIGHTS[3] + weekly_work_hours*WEIGHTS[4] + WEIGHTS[5])
	a = sigmoid(s)
	return a


def sigmoid(x):
	return 1 / (1 + math.exp(-x))