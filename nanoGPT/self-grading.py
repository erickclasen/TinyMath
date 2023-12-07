ROUNDING = 2
correct_count = 0
total_count = 0

with open('expressions.txt', 'r') as file:
    expressions = file.read().splitlines()

for expr in expressions:
    # Split the expression into the mathematical operation and the predicted result
    expr_parts = expr.split('=')
    
    # Check if there is an predicted result
    if len(expr_parts) > 1:
        operation = expr_parts[0].strip()
        predicted_result = float(expr_parts[1].strip())
    else:
        operation = expr.strip()
        predicted_result = None

    # Remove everything to the right of '=' and leading/trailing whitespaces
    expr = operation

    try:
        result = round(eval(expr), ROUNDING)
        if predicted_result is not None:
            total_count += 1
            if result == predicted_result:
                correct_count += 1
                print(f"{operation} = {result}   Correct")
            else:
                print(f"{operation} = {result}   Incorrect, model predicted {predicted_result}")
        else:
            print(f"{operation} = {result}")
    except Exception as e:
        print(f"Error evaluating expression: {operation}, {e}")

# Calculate and print the percentage
if total_count > 0:
    percentage = (correct_count / total_count) * 100
    print(f"\nScore: {correct_count}/{total_count} ({percentage:.2f}%)")
else:
    print("\nNo expressions to evaluate.")

