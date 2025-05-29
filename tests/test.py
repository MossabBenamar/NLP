# Simple program to calculate area of a rectangle

def calculate_area(length, width):
    area = length * width
    return area

# Get user input
length = float(input("Enter the length: "))
width = float(input("Enter the width: "))

# Calculate and display result
result = calculate_area(lenght, width)  # Error: 'lenght' should be 'length'
print(f"The area of the rectangle is: {result}")