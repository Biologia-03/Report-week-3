# BÀI 1: Cài đặt hàm Gaussian với input là 1 sốsố

import random
import numpy as np
import math


def gaussian(x, mean, std):
    # Gaussian formula
    coefficient = 1 / (std * math.sqrt(2 * math.pi))
    exponent = math.exp(-((x - mean) ** 2) / (2 * std ** 2))
    result = coefficient * exponent
    return round(result, 2)


# Input
x = 1
mean = 0
std = 1

# Test the function
result = gaussian(x, mean, std)
assert result == 0.24

print("Bài 1: ", gaussian(0.5, 1, 2))

# BÀI 2: Cài đặt hàm Gausian với input là 1 list


def gaussian_list(numbers, mean, std):
    results = []
    for x in numbers:
        # Gaussian formula
        coefficient = 1 / (std * math.sqrt(2 * math.pi))
        exponent = math.exp(-((x - mean) ** 2) / (2 * std ** 2))
        result = coefficient * exponent
        results.append(round(result, 2))  # Làm tròn tới 2 chữ số thập phân
    return results


# Input
numbers = [1, 2, 3]
mean = 0
std = 1

# Test the function
results = gaussian_list(numbers, mean, std)

assert results == [0.24, 0.05, 0.0]

numbers = [-3, -0.4, 2]
print("Bài 2: ", gaussian_list(numbers, 0, 2))

# BÀI 3: Cài đặt hàm Gausian với input là 1 numpy array


def gaussian_np(array, mean, std):
    # Gaussian formula áp dụng cho NumPy array
    coefficient = 1 / (std * np.sqrt(2 * np.pi))
    exponent = np.exp(-((array - mean) ** 2) / (2 * std ** 2))
    result = coefficient * exponent
    return np.round(result, 2)


# Input
array = np.array([0.1, 0.2, -3, 2, 5])
mean = 0
std = 1

# Test the function
results = gaussian_np(array, mean, std)
assert (results == np.array([0.4, 0.39, 0.0, 0.05, 0.0])).all()

array = np.array([-0.1, 0.01, -3])

print("Bài 3: ", gaussian_np(array, 0, 3))


# BÀI 4:


def generate_dice_rolls(n, seed=0):
    """Generate a list of dice rolls for an n-sided die, with a fixed seed for reproducibility."""
    random.seed(seed)
    return [random.randint(1, 6) for _ in range(n)]


def count_occurrences(dice_rolls, number):
    """Count the occurrences of a specific number in the list of dice rolls."""
    return dice_rolls.count(number)


# Example usage
n_rolls = 1000  # Number of dice rolls
dice_rolls = generate_dice_rolls(n_rolls)


# Count occurrences of number 1 and calculate its probability
number_of_interest = 6
occurrences = count_occurrences(dice_rolls, number_of_interest)

print("Bài 4:", occurrences)

# BÀI 5:


def calculate_probability(dice_rolls, number):
    """Calculate the probability of a specific number based on the dice rolls."""
    occurrences = count_occurrences(dice_rolls, number)
    return occurrences / len(dice_rolls)


# Example usage
n_rolls = 1000  # Number of dice rolls
dice_rolls = generate_dice_rolls(n_rolls)

# Count occurrences of number 4 and calculate its probability
number_of_interest = 4
probability = calculate_probability(dice_rolls, number_of_interest)

print("Bài 5:", probability)
