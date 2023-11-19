import random

def generate_math_dataset(filename, num_examples):
    with open(filename, 'w') as file:
        for _ in range(num_examples):
            num1 = random.randint(0, 90)
            num2 = random.randint(1, 90)  # Ensure num2 is not zero for division

            operation = random.choice(['+', '-', '*', '/'])
            result = 0

            if operation == '+':
                result = num1 + num2
            elif operation == '-':
                result = num1 - num2
            elif operation == '*':
                result = num1 * num2
            elif operation == '/':
                result = round(num1 / num2, 2)

            equation = f"{num1} {operation} {num2} = {result:.2f}\n"
            file.write(equation)

if __name__ == "__main__":
    output_filename = "math_dataset.txt"
    num_examples = 10000  # Change this to the desired number of examples

    generate_math_dataset(output_filename, num_examples)
    print(f"Math dataset '{output_filename}' with {num_examples} examples has been generated.")
