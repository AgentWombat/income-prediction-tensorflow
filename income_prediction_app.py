import model

# This is a small app which allows us to quickly test the moddel we have trained.

print("Welcome to the income predictor 5000!")
print("Answer a few questions and I will give you a prediction on" +
	" whether you make more than fifty-thousand dollars annually or not.")

while True:

	age = int(input("What is your age? (integer): "))
	has_college_degree = True if input("Do you have a college degree? (y/n): ").lower() == 'y' else False
	is_married = True if input("Are you married? (y/n): ").lower() == 'y' else False
	is_male = True if input("What is your sex? (male/female): ").lower() == 'male' else False
	weekly_work_hours = int(input("How many hours do you work each week? (integer): "))

	prediction = 100*model.predict(age, has_college_degree, is_married, is_male, weekly_work_hours)

	print(f"\nI am {prediction:.2f}% confident that you make over fifty-thousand dollars each year.")
	print("\n--#-$-#-$-#-$-#--\n")
