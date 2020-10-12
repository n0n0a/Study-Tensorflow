import numpy as np
import tensorflow as tf
import keras

class Company:
    def __init__(self,sales,cost,persons):
        self.sales = sales
        self.cost = cost
        self.persons = persons

    def get_profit(self):
        return self.sales -self.cost

if __name__ == '__main__':
    ca = Company(100,200,300)
    print(ca.get_profit())

    a = np.arange(6).reshape(2,3)
    a_m = np.mat(a)
    b = np.arange(6).reshape(3,2)
    b_m = np.mat(b)
    print(np.dot(a,b))
    print(a_m*b_m)
