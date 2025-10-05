# Level 1: Task 2
# Task: Temperature Conversion
def celsius_to_fahrenheit(celsius):
    return (celsius * 9/5) + 32

def fahrenheit_to_celsius(fahrenheit):
    return (fahrenheit - 32) * 5/9

def temperature_converter():
    print("--- Temperature Conversion Program ---")
    while True:
        try:
            value = float(input("Enter the temperature value: "))
            break
        except ValueError:
            print("Invalid input. Please enter a numerical value.")

    while True:
        unit = input("Enter the unit of measurement (C for Celsius, F for Fahrenheit): ").upper()
        if unit in ('C', 'F'):
            break
        else:
            print("Invalid unit. Please enter 'C' or 'F'.")

    if unit == 'C':
        converted_value = celsius_to_fahrenheit(value)
        print(f"\n{value}°C is equal to {converted_value:.2f}°F")
    elif unit == 'F':
        converted_value = fahrenheit_to_celsius(value)
        print(f"\n{value}°F is equal to {converted_value:.2f}°C")

if __name__ == "__main__":
    temperature_converter()
