results = []

ROUNDING = 2 

with open('expressions.txt', 'r') as file:
    expressions = file.read().splitlines()

for expr in expressions:
    # Remove everything to the right of '=' and leading/trailing whitespaces
    expr_left_side = expr.split('=')[0].strip()
    
    try:
        result = eval(expr_left_side)
        results.append((expr, round(result, ROUNDING)))
    except Exception as e:
        print(f"Error evaluating expression: {expr_left_side}, {e}")

for expr, result in results:
    print(f"{expr} = {result}")
