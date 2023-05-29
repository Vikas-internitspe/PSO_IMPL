


def factorial(n):
    '''Function for calculating the factorial for a number'''
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result

def is_prime(n):
    '''Function for checking the given number is prime on not'''
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True