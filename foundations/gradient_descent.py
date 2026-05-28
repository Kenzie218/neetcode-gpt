class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: float) -> float:
        x = init
        for i in range(iterations):
            #gradient is the derivative
            gradient = 2 * x
            x = x - (learning_rate * gradient)
        
        return round(x, 5)