# ----------Learning Python------------

# --------Day 1(Python) Date:15/03/23-------------
import random
import mymodule
import platform
import datetime
import math
from math import log
import json
import re
import os
import scipy
from scipy.spatial.distance import hamming
from scipy import constants
import pandas as pd
import numpy as np
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import ConvexHull
from scipy.sparse.csgraph import dijkstra
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
from scipy.optimize import root
from scipy.optimize import minimize
from math import cos
from scipy.spatial.distance import cityblock
from scipy.spatial import KDTree
from scipy.sparse.csgraph import depth_first_order
from scipy.sparse.csgraph import breadth_first_order
from scipy.sparse.csgraph import bellman_ford
from scipy.sparse.csgraph import floyd_warshall
from scipy.spatial import Delaunay
from scipy import io
from scipy.interpolate import interp1d
from scipy.spatial.distance import cosine
from scipy.spatial.distance import euclidean
from scipy.interpolate import Rbf

from mymodule import person1


class AnanyasPythonPractice:
    print("-------------------------------Day 1(Python) Date:16/03/23-------------------------")


# region HelloWorld!
print("    ")

print("Hello World")

if (5 > 2):
    print("5 is greater than 2")
else:
    print("5 is not greater than 2")

# endregion

# region variables
# ------------variables-------------
print("--------------------------------Variables----------------------------------------")
print("     ")
print("[Note:Python has no command for declaring a variable]")
i = 5
j = "Hello world"
print("Printing an integer using variable:")
print(i)
print("Printing a string using variable:")
print(j)

# ------------------Output variables---------------------

print("--------------------Output variables---------------------")

x = "Python is awesome"
print(x)

x = "Python"
y = "is"
z = "awesome"
print(x, y, z)

x = "Python "
y = "is "
z = "awesome"
print(x + y + z)

x = 5
y = 10
print(x + y)

"""
x = 5
y = "John"
print(x + y) //error cannot concatenate int with string

"""

x = 5
y = "John"
print(x, y)  # prints 5 John

# ----------------Global variables----------------

print("------------Using Global variables-----------------")

x = "awesome"


def myfunc():
    print("Python is " + x)


myfunc()

# --------------Using Local variables-----------------

print("------------Using Local variables-----------------")


def myfunc():
    x = "fantastic"
    print("Local variable : Python is " + x)


myfunc()

print("Global variable remains as awesome :python is " + x)


# ----------------Using Global Keyword-----------------------

def myfunc():
    global x
    x = "fantastic"
    print("Inside function(using global keyword) : Python is " + x)


myfunc()

print("Outside function(using global keyword) : Python is " + x)

# endregion

# region Comments
# ---------------Comments---------------------

"""
Practicing to comment
in more than
just one
line
"""
# endregion

# region casting
# -------------------casting-----------------

print("-----------------------------Casting-----------------------------------------")

x = str(3)  # x will be '3'
y = int(3)  # y will be 3
z = float(3)  # z will be 3.0

print(x)
print(y)
print(z)

# --------------------Get The type-----------------------------

print("-------------------Get the type of the variable------------------------")

print(type(x))
print(type(y))
print(type(z))

print("   ")

print("[Note: string variables can be declared using single or double quotes]")

"""
x = "John"
# is the same as
x = 'John'

"""
print("      ")
print("-----------Multi word variables----------------")

print("   ")

print("Camel Case: myVariableName")
print("Pascal Case: MyVariableName")
print("Snake  Case: my_variable_name")
print("    ")

print("----------Assigning Many Values to Multiple Variables[many-many]---------")

x, y, z = "Horse", 6, "Moulya"
print("[Make sure the number of variables matches the number of values or else you'll get an error]")
print(x)
print(y)
print(z)

print("     ")

print("----------Assigning one Value to Multiple Variables[many-one]---------")

x = y = z = 10
print(x)
print(y)
print(z)

# endregion

# region unpacking
# ----------------Unpacking-------------------
print("----------------------Unpack a collection(list here)---------------")

fruits = ["apple", "banana", "cherry"]
x, y, z = fruits

print(x)
print(y)
print(z)
# endregion

# region Printing Random Numbers
# -----------------Printing Random Numbers---------------------

print("  ")
print("---------------------------Printing random numbers in  given range-------------------------")
print("Note : {Python does not have a random() function to make a random number.")
print("        But Python has a built-in module called random that can be used to make random numbers}:")
print("    ")
print("Random number is:", random.randrange(1, 10))

# endregion

# region Checking if a string or a character is present or not!
# --------------------Checking if a string or a character is present or not!-----------------------
print("      ")
print("---------------------Checking  if a string or a character is present or not!-------------------------")
print("Note : To check if a certain phrase or character is present in a string, we can use the keyword in.")

txt = "The best things in life are free!"

print("free" in txt)  # Returns True

print("----------------Using (in) in if stmt---------------------")

if "free" in txt:
    print("Yes, 'free' is present.")

print("----------------------Return the length of the string-----------------------")

a = "Hello, World!"
print("Hello, World! ", len(a))  # Returns 13

# ---------------------------Check if not(check if the string is not present--------------------------

print(
    "--------------- To check if a certain phrase or character is NOT present in a string, we can use the keyword not in. -------------------------------- ")
# not in
print("expensive" not in txt)  # Returns true

print("----------------Using (not in) in if stmt---------------------")
if "expensive" not in txt:
    print("No, 'expensive' is NOT present.")

# endregion

# region String Concatenation
# ----------------------------String Concatenation----------------------------

print("---------------------------String Concatenation----------------------------")
quantity = 3
itemno = 567
price = 49.95
myorder = "I want {} pieces of item {} for {} dollars."
print(myorder.format(quantity, itemno, price))

print("Note: You can use numbers inside placeholders to rearrange the variable positions..!")
quantity = 3
itemno = 567
price = 49.95
myorder = "I want {2} pieces of item {1} for {0} dollars."
print(myorder.format(quantity, itemno, price))

# endregion

# region Passing a list as an argument
print("   ")
print("-------------------------------Day 2(Python) Date:17/03/23-------------------------")
print("    ")
# -----------------------Passing a list as an argument--------------------------

print("----------------------Passing a list as an argument-----------------")


def my_function(food):
    for x in food:
        print(x)


fruits = ["Banana", "Orange", "Apple"]
my_function(fruits)
# endregion

# region Returning values
# -----------------------Returning values--------------------------

print(
    "----------------------Returning values inside a function and printing the function by calling it with parameter-----------------")


def my_function(x):
    return 5 * x


print(my_function(3))
print(my_function(5))
print(my_function(9))

# endregion

# region Iterators
# ----------------------------------Iterators------------------------------
print("--------------------Iterators------------------------------")
mytuple = ("apple", "banana", "cherry")
myit = iter(mytuple)

print(next(myit))
print(next(myit))
print(next(myit))

mystr = "banana"
myit = iter(mystr)

print(next(myit))
print(next(myit))
print(next(myit))
print(next(myit))
print(next(myit))
print(next(myit))

# endregion

# region Importing modules
# ----------------------------Importing modules---------------------------
print("----------------------------Importing modules---------------------------")

mymodule.greeting("Ananya H S")

a = mymodule.person1["age"]
print(a)
# -----------------------Another method to print---------------
print(person1["name"])
print(person1["country"])

# ----------------------------To check which operating system is it----------------

print("--------------Using Build-in platform module to check which is the operating system----------------")
x = platform.system()
print(x)

# ---------------------------Using the dir() Function------------------------
print("          ")
print("Note : A dir() function is a built-in function to list all the function names (or variable names) in a module.")
print("       ")
x = dir(mymodule)
print(x)
# endregion

# region DateTime

# ----------------------------DateTime---------------------------------------

print("    ")
print(
    "Note: A date in Python is not a data type of its own, but we can import a module named datetime to work with dates as date objects.")
x = datetime.datetime.now()
print("Current date and time")
print(x)

print(
    "Printing the month,year,and day of my birthday...!(Using x.strftime(%B) for month) and (x.year) for year and  (x.strftime(%A)) for day")
x = datetime.datetime(2001, 8, 6)  # year, month, date
print(x.strftime("%B"))
print(x.year)
print(x.strftime("%A"))
# endregion

# region math Functions
# ------------------------------Math functions----------------------------------
print("----------------------Implementing math functions------------------------")
x = min(5, 10, 25)
y = max(5, 10, 25)
z = abs(-7.5)  # Returns +7.5
q = pow(4, 3)  # Returns 4^3 => 4*4*4
s = math.sqrt(64)
w = math.ceil(1.4)
f = math.floor(1.4)
u = math.pi

print("Minimum of 5,10,25")
print(x)
print("Maximum of 5,10,25")
print(y)
print("Absolute of -7.5")
print(z)
print("Power of 4^3")
print(q)
print("Squareroot of 64")
print(s)
print("Print ceil,rounds off to max")
print(w)
print("Print floor,rounds off to min")
print(f)
print("Print the value of pi")
print(u)

# endregion

# region JSON to PYTHON
print("   ")
print("-------------------------------Day 3(Python) Date:20/03/23-------------------------")
print("    ")

# some JSON:
x = '{ "name":"John", "age":30, "city":"New York"}'

# parse x:
y = json.loads(x)

print("--------------------------------- Convert from JSON to Python------------------------------")
# the result is a Python dictionary:
print(y["age"])
print("    ")

# a Python object (dict):
x = {
    "name": "John",
    "age": 30,
    "city": "New York"
}

# convert into JSON:
y = json.dumps(x)
# endregion

# region Python to JSON
print("--------------- Convert from Python to JSON------------------------------")
# the result is a JSON string:
print(y)
print("    ")

# endregion

# region Python RegEx
print("-----------------Python RegEx----------------------------")
# Check if the string starts with "The" and ends with "Spain":

txt = "The rain in Spain"
print(txt)
print("      ")
x = re.search("^The.*Spain$", txt)
if x:
    print("YES! We have a match!")
else:
    print("No match")

# endregion

# region LocationOfPythonAndPIP
print("    ")

print("Location of python : C:\ Users\Administrator\AppData\Local\Programs\Python\Python310\python.exe")
print("     ")
print(
    "Command to check PIP Version : C:\ Users\Administrator\AppData\Local\Programs\Python\Python310\Scripts\pip --version ")
print("      ")
print(
    "PIP VERSION : pip 22.3.1 from C:\ Users\Administrator\AppData\Local\Programs\Python\Python310\lib\site-packages\pip (python 3.10)")
print("    ")

# endregion

# region Exception Handling
print("----------------------------Exception Handling------------------------------")

# The try block will generate a NameError, because x is not defined:

print("Using else in Exception handling")
try:
    print(2 / 0)
except:
    print("Something went wrong")
else:
    print("Nothing went wrong")

# The "finally" block gets executed no matter if the try block raises any errors or not:


print("Using finally in exception Handling")
try:
    print(k)
except:
    print("Something went wrong")
finally:
    print("The 'try except' is finished")

"""
#Raising Exceptions:

print("-----------------Raising Exceptions------------------------")

x = -1

if x < 0:
    raise Exception("Sorry, no numbers below zero")

"""

# endregion

# region User Input
print("-------------------User Input------------------")
print("      ")
print("Python User Input")

username = input("Enter username:")
print("Username is: " + username)

# endregion

# region Python file handling
print("------------------Python file handling------------------")

f = open("demofile.txt", "r")
print(f.read())

print("  ")

f = open("demofile.txt", "r")
for x in f:
    print(x)
f.close()

# endregion

# region Numpy [Python Library]
print("------------------------------Numpy [Python Library]--------------------------------")

arr = np.array([1, 2, 3, 4, 5])

print(arr)
print("Checking the type of array: ")
print(type(arr))

print("Checking numpy's version")
print(np.__version__)

print("Checking number of dimensions")
a = np.array(42)
b = np.array([1, 2, 3, 4, 5])
c = np.array([[1, 2, 3], [4, 5, 6]])
d = np.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])

print(a.ndim)
print(b.ndim)
print(c.ndim)
print(d.ndim)

print("   ")
print("-------------------------------Day 4(Python) Date:24/03/23-------------------------")
print("    ")

# ---------------------Generate Random Number-------------------------

print("------------------Generate Random Number(integer)---------------")

x = random.randint(0, 100)

print(x)

print("------------------Generate Random Number(float)---------------")

x = np.random.rand()

print(x)

print("------------------Generate random array of size=5-----------------")
x = np.random.randint(100, size=(5))

print(x)

print("------------------printing an array of defined set of integers with probability and size values-----------")

x = np.random.choice([3, 5, 7, 9], p=[0.1, 0.3, 0.6, 0.0], size=(100))

print(x)

print("------------------Shuffling an array using shuffle() method------------------")
print("Note:The shuffle() method makes changes to the original array.")

arr = np.array([1, 2, 3, 4, 5])
random.shuffle(arr)
print(arr)

print("---------------------Original Array---------------------")
print(arr)

print("----------------------------Shuffling an array using permutation() method-------------------")
print("Note:The permutation() method returns a re-arranged array (and leaves the original array un-changed).")

arr = np.array([1, 2, 3, 4, 5])
z = np.random.permutation(arr)
print(z)

print("---------------------Original Array---------------------")
print(arr)

# region SeabornModule
"""
print("--------------Implementing seaborn module----------------")
print("Note : Seaborn is a library that uses Matplotlib underneath to plot graphs. It will be used to visualize random distributions.")

print("------------------Plotting a Distplot----------------")
sns.distplot([0, 1, 2, 3, 4, 5])
plt.show()

print("   ")
print("-------------------------------Day 5(Python) Date:28/03/23-------------------------")
print("    ")


print("----------------Plotting a Distplot Without the Histogram------------------")
sns.distplot([0, 1, 2, 3, 4, 5], hist=False)
plt.show()

print("Generate a random normal distribution of size 2x3 with mean(loc) at 1 and standard deviation(scale) of 2:")

x = np.random.normal(loc=1, scale=2, size=(2, 3))
print(x)

print("-----------------------Visualization of Normal Distribution----------------------")

sns.distplot(np.random.normal(size=1000), hist=False)
plt.show()

print("-----------------Binomial Distruibution----------------")

x = np.random.binomial(n=10, p=0.5, size=10)

print(x)

print("--------------------Visualization of Binomial Distribution----------------------")
sns.distplot(np.random.binomial(n=10, p=0.5, size=1000), hist=True, kde=False)
plt.show()

print("------------------Difference Between Normal and Binomial Distribution-------------")
print("Note: The main difference is that normal distribution is continous whereas binomial is discrete, but if there are enough data points it will be quite similar to normal distribution with certain loc and scale.")

sns.distplot(np.random.normal(loc=50, scale=5, size=1000), hist=False,label='normal')
sns.distplot(np.random.binomial(n=100, p=0.5, size=1000), hist=False, label='binomial')
plt.show()

print("------------------------Poisson Distribution------------------")

x = np.random.poisson(lam=2, size=10)

print(x)

print("-----------------------------Visualization of Poisson Distribution----------------------")
sns.distplot(np.random.poisson(lam=2, size=1000), kde=False)
plt.show()

print("----------------------Difference Between Normal and Poisson Distribution-----------------")


sns.distplot(np.random.normal(loc=50, scale=7, size=1000), hist=False, label='normal')
sns.distplot(np.random.poisson(lam=50, size=1000), hist=False, label='poisson')
plt.show()

print("------------------------Difference Between Binomial and Poisson Distribution-----------------")

sns.distplot(np.random.binomial(n=1000, p=0.01, size=1000), hist=False, label='binomial')
sns.distplot(np.random.poisson(lam=10, size=1000), hist=False, label='poisson')
plt.show()

print("---------------------Uniform Distribution-------------------------")
x = np.random.uniform(size=(2, 3))
print(x)

print("------------------------Visualization of Uniform Distribution----------------------")
sns.distplot(np.random.uniform(size=1000), hist=False)
plt.show()

print("-----------------------Logistic Distribution----------------------------")
x = np.random.logistic(loc=1, scale=2, size=(2, 3))
print(x)

print("-------------------------Visualization of Logistic Distribution-----------------------")
sns.distplot(np.random.logistic(size=1000), hist=False)
plt.show()

print("---------------------------Difference Between Logistic and Normal Distribution-------------")
sns.distplot(np.random.normal(scale=2, size=1000), hist=False, label='normal')
sns.distplot(np.random.logistic(size=1000), hist=False, label='logistic')
plt.show()

print("---------------------------Multinomial Distribution----------------------------")
x = np.random.multinomial(n=6, pvals=[1/6, 1/6, 1/6, 1/6, 1/6, 1/6])
print(x)

print("-------------------Exponential distribution--------------------")
x = np.random.exponential(scale=2, size=(2, 3))
print(x)

print("---------------------Visualization of Exponential Distribution----------------")
sns.distplot(np.random.exponential(size=1000), hist=False)
plt.show()

print("---------------------Chi Square Distribution---------------------------")
x = np.random.chisquare(df=2, size=(2, 3))
print(x)

print("---------------------Visualization of Chi Square Distribution----------------")
sns.distplot(np.random.chisquare(df=1, size=1000), hist=False)
plt.show()

print("----------------------Rayleigh Distribution-------------------------")
x = np.random.rayleigh(scale=2, size=(2, 3))
print(x)

print("----------------------Visualization of Rayleigh Distribution-----------------")
sns.distplot(np.random.rayleigh(size=1000), hist=False)
plt.show()

print("----------------------------Pareto Distribution-----------------------------")
x = np.random.pareto(a=2, size=(2, 3))
print(x)

print("-------------------------Visualization of Pareto Distribution-----------------")

sns.distplot(np.random.pareto(a=2, size=1000), kde=False)
plt.show()

print("-------------------Zipf Distribution-----------------")
x = np.random.zipf(a=2, size=(2, 3))
print(x)

print("------------------Visualization of Zipf Distribution--------------------------")
x = np.random.zipf(a=2, size=1000)
sns.distplot(x[x<10], kde=False)
plt.show()

"""
# endregion

# endregion

# region pythonUfunc(Universal Functions)

print("   ")
print("-------------------------------Day 6(Python) Date:29/03/23-------------------------")
print("    ")

print("-----------------------------Python Universal functions(ufuncs)----------------------")
# region Vectorization
print("        ")
print("----------------------------Vectorization-----------------------------------")
print("           ")
print("Note:Converting iterative statements into a vector based operation is called vectorization.")
print("        ")
print("---------------------Adding the Elements of two Lists using python's built in zip() method--------------------")
print("list 1: [1, 2, 3, 4]")
print("list 2: [4, 5, 6, 7]")

x = [1, 2, 3, 4]
y = [4, 5, 6, 7]
z = []

for i, j in zip(x, y):
    z.append(i + j)
print("After adding two lists :", z)

print("Note:NumPy has a ufunc for this, called add(x, y) that will produce the same result.")

z = np.add(x, y)
print("After adding two lists :", z)

# endregion

# region Create Your Own ufunc
print("     ")
print(" --------------------Create Your Own ufunc------------------")


def myadd(x, y):
    return x + y


myaddobj = np.frompyfunc(myadd, 2, 1)

print(myaddobj([1, 2, 3, 4], [5, 6, 7, 8]))

print("     ")

print("-----------------Check the type of a function to check if it is a ufunc or not.-----------")
print(type(np.add))

print("    ")
print(" --------------------Check the type of another function: concatenate():----------------------- ")
print(type(np.concatenate))

# endregion

# region Simple arithmetic
print("------------------------Simple arithmetic ---------------------")
print("  ")
print("                    Addition(using add ufunc)                       ")
print("Note: The add() function sums the content of two arrays, and return the results in a new array.")

arr1 = np.array([10, 11, 12, 13, 14, 15])
arr2 = np.array([20, 21, 22, 23, 24, 25])
newarr = np.add(arr1, arr2)
print(newarr)

print("                      Subtraction(uising subtract ufunc)                  ")
print(
    "Note: The subtract() function subtracts the values from one array with the values from another array, and return the results in a new array.")

arr1 = np.array([10, 11, 12, 13, 14, 15])
arr2 = np.array([20, 21, 22, 23, 24, 25])
newarr = np.subtract(arr1, arr2)
print(newarr)

print("-------------------------Rounding Decimals-----------------------")
print("Using trunc() method to remove decimals in a floating point number")
arr = np.trunc([-3.1666, 3.6667])
print(arr)

print("         ")
print("Using fix() method to remove decimals in a floating point number")
arr = np.fix([-3.1666, 3.6667])
print(arr)

print("         ")
print("The around() function increments preceding digit or decimal by 1 if >=5 else do nothing.")
print("E.g. round off to 1 decimal point, 3.16666 is 3.2")

print("       ")
print("Round off 3.1666 to 2 decimal places:")
arr = np.around(3.1666, 2)
print(arr)

print("        ")
# endregion

# region Numpy Logs
print("---------------------------------------Numpy Logs--------------------------------")
print("     ")
print("---------------------------Log at base 2------------------")
arr = np.arange(1, 10)
print(np.log2(arr))
print("       ")

print("--------------------------Log at Base 10--------------------")
arr = np.arange(1, 10)
print(np.log10(arr))

print("-----------------------Log at any base--------------------")
nplog = np.frompyfunc(log, 2, 1)
print(nplog(100, 15))
# endregion

# region numpy summations
print("---------------------Print numpy summations-------------------")
arr1 = np.array([1, 2, 3])
arr2 = np.array([1, 2, 3])

newarr = np.sum([arr1, arr2])
print(newarr)

print("    ")

print("-------------------Print numpy summations over an axis------------")
newarr = np.sum([arr1, arr2], axis=1)
print(newarr)

print("---------------------Cummulative sum-------------------")
arr = np.array([1, 2, 3])
newarr = np.cumsum(arr)
print(newarr)

# endregion

# region numpy products

print("         ")
print("--------------------Numpy products----------------")
print("   ")
print("product of entire array[1,2,3,4]")
arr = np.array([1, 2, 3, 4])
x = np.prod(arr)
print(x)

print("      ")

print("product of 2 arrays:[1,2,3,4][5,6,7,8]")
arr1 = np.array([1, 2, 3, 4])
arr2 = np.array([5, 6, 7, 8])
x = np.prod([arr1, arr2])
print(x)

print("     ")

print("-------------------------Product Over an Axis----------------------------")
print("    ")
print("Note:If you specify axis=1, NumPy will return the product of each array.")

newarr = np.prod([arr1, arr2], axis=1)
print(newarr)

print("---------------------------Cummulative product-------------------------")
print("     ")

print("Note: Cummulative product means taking the product partially.")
print("      E.g. The partial product of [1, 2, 3, 4] is [1, 1*2, 1*2*3, 1*2*3*4] = [1, 2, 6, 24]")
print("      Perfom partial sum with the cumprod() function.")

print("     ")

arr = np.array([5, 6, 7, 8])
newarr = np.cumprod(arr)
print(newarr)
# endregion

# region Numpy Differences
print("---------------------------Numpy Differences----------------------------")

print("Note:    A discrete difference means subtracting two successive elements.")
print("         E.g. for [1, 2, 3, 4], the discrete difference would be [2-1, 3-2, 4-3] = [1, 1, 1]")
print("         To find the discrete difference, use the diff() function.")

print("      ")

print("Printing discrete difference: ")
arr = np.array([10, 15, 25, 5])
newarr = np.diff(arr)
print(newarr)

print("      ")

print("To repeat the discrete difference operation give the value for n:")
arr = np.array([10, 15, 25, 5])
newarr = np.diff(arr, n=2)
print(newarr)

# endregion

# region LCM

print("-------------------Finding LCM (Lowest Common Multiple)---------------------")
num1 = 4
num2 = 6

x = np.lcm(num1, num2)
print(x)

print("     ")

print("-------------------Finding LCM in Arrays-----------------------")

arr = np.array([3, 6, 9])
x = np.lcm.reduce(arr)
print(x)

print("     ")

print("Find the LCM of all values of an array where the array contains all integers from 1 to 10:")
arr = np.arange(1, 11)
x = np.lcm.reduce(arr)
print(x)

print("      ")

# endregion

# region GCD

print("-------------------Finding GCD (Greatest Common Denominator)--------------------")

num1 = 20
num2 = 30

x = np.gcd(num1, num2)
print(x)

print("--------------------Finding GCD in Arrays------------------")

arr = np.array([20, 8, 32, 36, 6])
x = np.gcd.reduce(arr)
print(x)

print("       ")
# endregion

# region NumPy Trigonometric Functions
print("----------------------NumPy Trigonometric Functions-------------------")

x = np.sin(np.pi / 2)
print("sin90:")
print(x)

print("-----------------------Find sine values for all of the values in arr:---------------")

arr = np.array([np.pi / 2, np.pi / 3, np.pi / 4, np.pi / 5])
x = np.sin(arr)
print(x)

print("   ")

print("---------------------------Convert degrees into radians---------------------")
arr = np.array([90, 180, 270, 360])
x = np.deg2rad(arr)
print(x)

print("     ")

print("------------------------Convert radians to degrees--------------------")
arr = np.array([np.pi / 2, np.pi, 1.5 * np.pi, 2 * np.pi])
x = np.rad2deg(arr)
print(x)

print("     ")

print("----------------------Finding Angles------------------------")

x = np.arcsin(1.0)
print(x)

print("       ")

print("------------------------Angles of Each Value in Arrays ----------------------")
arr = np.array([1, -1, 0.1])
x = np.arcsin(arr)
print(x)

print("      ")

print(" ------------------Hypotenues-------------------------")
base = 3
perp = 4
x = np.hypot(base, perp)
print(x)

print("     ")

print("-------------------Numpy hyperbolioc functions---------------")
x = np.sinh(np.pi / 2)
print(x)

print("       ")

print("-------------Find cosh(cos hyperbolic) values for all of the values in arr:-----------")
arr = np.array([np.pi / 2, np.pi / 3, np.pi / 4, np.pi / 5])
x = np.cosh(arr)
print(x)

print("       ")
# endregion

# region setoperations
print("-------------Numpy set Operations--------------")

print("----------Convert following array with repeated elements to a set:----------")

arr = np.array([1, 1, 1, 2, 3, 4, 5, 5, 6, 7])
x = np.unique(arr)
print(x)

print("    ")

print("--------------Finding Union------------------")
arr1 = np.array([1, 2, 3, 4])
arr2 = np.array([3, 4, 5, 6])

newarr = np.union1d(arr1, arr2)
print(newarr)

print("    ")

print("--------------Finding Intersection------------")

newarr = np.intersect1d(arr1, arr2, assume_unique=True)
print(newarr)

print("   ")

print("----------Finding Difference-------------")

newarr = np.setdiff1d(arr1, arr2, assume_unique=True)
print(newarr)

print("-----------Finding Symmetric Difference-----------")

newarr = np.setxor1d(arr1, arr2, assume_unique=True)
print(newarr)

# endregion

# endregion

# region Pandas [Python Library]


print("------------------------------Pandas [Python Library]--------------------------------")

mydataset = {
    'cars': ["BMW", "Volvo", "Ford"],
    'passings': [3, 7, 2]
}

myvar = pd.DataFrame(mydataset)
print(myvar)

print(" ")

print("-----------------------Checking pandas version-------------------")
print(pd.__version__)

print("   ")

print(" ----------------------Pandas Series----------------------")
a = [1, 7, 2]

myvar = pd.Series(a)
print(myvar)

print(
    "Note : If nothing else is specified, the values are labeled with their index number. First value has index 0, second value has index 1 etc.")
print("       This label can be used to access a specified value.")

print(myvar[0])

print(
    "--------After manual indexing the array with x,y,z ,accessing the elements using the created index--------------------")
myvar = pd.Series(a, index=["x", "y", "z"])

print(myvar["y"])

print("------------Passing key:value objects as Series----------------")

calories = {"day1": 420, "day2": 380, "day3": 390}
myvar = pd.Series(calories, index=["day1", "day2"])

print(myvar)

print("------------Dataframes----------------")

data = {
    "calories": [420, 380, 390],
    "duration": [50, 40, 45]
}

df = pd.DataFrame(data)
print(df)

print("-----------To return one/more rows loc attribute is used--------------")
print("Returning a single row: ")
print(df.loc[0])
print("Note: This example returns a Pandas Series.")

print("  ")

print("Returning more than 1 row : ")
print(df.loc[[0, 1]])
print("Note: When using [], the result is a Pandas DataFrame.")

print("  ")

print("--------------Named indexes-----------------")

df = pd.DataFrame(data, index=["day1", "day2", "day3"])
print(df)

print("------------Locate Named Indexes--------------")

print(df.loc["day2"])

print("-----------Load files into a DataFrame---------------")
df = pd.read_csv('data.csv')
print(df)
print(
    "Note : If you have a large DataFrame with many rows, Pandas will only return the first 5 rows, and the last 5 rows")

print("         ")
print("-----------------Read CSV Files------------------")

df = pd.read_csv('data.csv')
print(df.to_string())

print("  ")
print("Displaying the maximum number of rows this computer can display in a dataframe,! ")
print(pd.options.display.max_rows)

print("  ")
print("-----------------------------Pandas Read JSON----------------------------------")

df = pd.read_json('data.json')
print(df.to_string())
print("   ")

print(
    "-----------------------------Pandas Dictionary as  JSON to print directly as a dataframe----------------------------------")

data = {
    "Duration": {
        "0": 60,
        "1": 60,
        "2": 60,
        "3": 45,
        "4": 45,
        "5": 60
    },
    "Pulse": {
        "0": 110,
        "1": 117,
        "2": 103,
        "3": 109,
        "4": 117,
        "5": 102
    },
    "Maxpulse": {
        "0": 130,
        "1": 145,
        "2": 135,
        "3": 175,
        "4": 148,
        "5": 127
    },
    "Calories": {
        "0": 409,
        "1": 479,
        "2": 340,
        "3": 282,
        "4": 406,
        "5": 300
    }
}
df = pd.DataFrame(data)
print(df)

print("-----------------------Analysing dataframes---------------------")

print("    ")

print("-----------Viewing the Data--------------")
df = pd.read_csv('data.csv')

print("Using head() method to return first 5 rows by default")
print(df.head())
print("       ")

print("Using head() method to return specified number of rows from first:")
print(df.head(10))
print("       ")

print("Using tail() method to return last 5 rows by default")
print(df.tail())
print("       ")

print("Using tail() method to return specified number of rows from last:")
print(df.tail(10))
print("       ")

print("-----------Info about the data-------------")
print(df.info())

print("---------------------------Pandas - Cleaning Data---------------------------")

print("------Pandas - Cleaning Empty Cells by removing rows containing empty cells----------")
df = pd.read_csv('data2.csv')

new_df = df.dropna()

print("Removing rows containing empty cells:")
print(new_df.to_string())

print("If you want to change the original DataFrame, use the inplace = True argument:")
df = pd.read_csv('data2.csv')
df.dropna(inplace=True)
print(df.to_string())

print("------------------------Replace values in selected columns--------------")
df = pd.read_csv('data2.csv')
df["Calories"].fillna(130, inplace=True)
print(df.to_string())

print("-----------------------Replace Empty Values---------------------")
df = pd.read_csv('data2.csv')
df.fillna(130, inplace=True)
print(df.to_string())

print("-----------------------Pandas - Cleaning Data of Wrong Format-----------------")
df = pd.read_csv('data2.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.dropna(subset=['Date'], inplace=True)  # to remove empty rows
print(df.to_string())

print("-----------------------Pandas - Fixing Wrong Data---------------------------")
print("    ")
print("By replacing values:")
print("    ")
df = pd.read_csv('data2.csv')
df.loc[7, 'Duration'] = 45
df['Date'] = pd.to_datetime(df['Date'])
print(df.to_string())
print(
    "To replace wrong data for larger data sets you can create some rules, e.g. set some boundaries for legal values, and replace any values that are outside of the boundaries.")

for x in df.index:
    if df.loc[x, "Duration"] > 120:
        df.loc[x, "Duration"] = 120

print(df.to_string())

print("Removing the rows containing wrong data:")
for x in df.index:
    if df.loc[x, "Duration"] < 45:
        df.drop(x, inplace=True)

print(df.to_string())

print("-------------------------Removing Duplicates----------------------")

print(
    "See where duplicates are there : False : Unique Value , True : Duplicate/Repeated Value using the duplicated() method:")
df = pd.read_csv('data2.csv')
print(df.duplicated())

print("To remove duplicates, use the drop_duplicates() method.")
df = pd.read_csv('data2.csv')
df.drop_duplicates(inplace=True)
print(df.to_string())

print("--------------------------Pandas - Data Correlations----------------------")
df = pd.read_csv('data.csv')
print(df.corr())

print("---------------------------Pandas - Plotting-------------------------------")

print("Plotting:")

df = pd.read_csv('data.csv')
df.plot()
plt.show()

print("-----------Scatter plot----------")
df = pd.read_csv('data.csv')
print("Scatter plot wih good correlation:0.922721")
df.plot(kind='scatter', x='Duration', y='Calories')
plt.show()

print("Another scatter plot with bad correlation :0.0093657")
df = pd.read_csv('data.csv')
df.plot(kind='scatter', x='Duration', y='Maxpulse')
plt.show()

print("-------------Histogram-------------")
df = pd.read_csv('data.csv')
df['Duration'].plot(kind='hist')
plt.show()

print("Pandas Tutorial completed...!")
# endregion

# region Scipy

print("Printing Scipy Version :")
print(scipy.__version__)

# region Scipy Constants
print("How many cubic meters are in one liter:")
print(constants.liter)

print("Printing the value of pi:")
print(constants.pi)

print("A list of all units under the constants module can be seen using the dir() function:")
print(dir(constants))

print("Metric (SI) Prefixes:")
print("Note : Returns the specified unit in meter (e.g. centi returns 0.01)")
print(constants.yotta)  # 1e+24
print(constants.zetta)  # 1e+21
print(constants.exa)  # 1e+18
print(constants.peta)  # 1000000000000000.0
print(constants.tera)  # 1000000000000.0
print(constants.giga)  # 1000000000.0
print(constants.mega)  # 1000000.0
print(constants.kilo)  # 1000.0
print(constants.hecto)  # 100.0
print(constants.deka)  # 10.0
print(constants.deci)  # 0.1
print(constants.centi)  # 0.01
print(constants.milli)  # 0.001
print(constants.micro)  # 1e-06
print(constants.nano)  # 1e-09
print(constants.pico)  # 1e-12
print(constants.femto)  # 1e-15
print(constants.atto)  # 1e-18
print(constants.zepto)  # 1e-21

print("   ")

print("Binary Prefixes:")
print("Note : Returns the specified unit in bytes (e.g. kibi returns 1024)")
print(constants.kibi)  # 1024
print(constants.mebi)  # 1048576
print(constants.gibi)  # 1073741824
print(constants.tebi)  # 1099511627776
print(constants.pebi)  # 1125899906842624
print(constants.exbi)  # 1152921504606846976
print(constants.zebi)  # 1180591620717411303424
print(constants.yobi)  # 1208925819614629174706176

print("Mass : ")
print("Return the specified unit in kg (e.g. gram returns 0.001)")
print(constants.gram)  # 0.001
print(constants.metric_ton)  # 1000.0
print(constants.grain)  # 6.479891e-05
print(constants.lb)  # 0.45359236999999997
print(constants.pound)  # 0.45359236999999997
print(constants.oz)  # 0.028349523124999998
print(constants.ounce)  # 0.028349523124999998
print(constants.stone)  # 6.3502931799999995
print(constants.long_ton)  # 1016.0469088
print(constants.short_ton)  # 907.1847399999999
print(constants.troy_ounce)  # 0.031103476799999998
print(constants.troy_pound)  # 0.37324172159999996
print(constants.carat)  # 0.0002
print(constants.atomic_mass)  # 1.66053904e-27
print(constants.m_u)  # 1.66053904e-27
print(constants.u)  # 1.66053904e-27

print("   ")

print("Angle :")
print("Return the specified unit in radians (e.g. degree returns 0.017453292519943295)")
print(constants.degree)  # 0.017453292519943295
print(constants.arcmin)  # 0.0002908882086657216
print(constants.arcminute)  # 0.0002908882086657216
print(constants.arcsec)  # 4.84813681109536e-06
print(constants.arcsecond)  # 4.84813681109536e-06

print("Time:  ")
print("Return the specified unit in seconds (e.g. hour returns 3600.0)")
print(constants.minute)  # 60.0
print(constants.hour)  # 3600.0
print(constants.day)  # 86400.0
print(constants.week)  # 604800.0
print(constants.year)  # 31536000.0
print(constants.Julian_year)  # 31557600.0

# endregion

# region Scipy Optimizers
print("   ")

print("----------------------SciPy Optimizers----------------------")

print("Printing the roots of an equation(x + cos(x)):")


def eqn(x):
    return x + cos(x)


myroot = root(eqn, 0)
print(myroot.x)
print("Extra Info:", myroot)

print("   ")

print("Minimize the function x^2 + x + 2 with BFGS:")


def eqn(x):
    return x ** 2 + x + 2


mymin = minimize(eqn, 0, method='BFGS')
print(mymin.x)
print("Extra Info:", mymin)

# endregion

# region Scipy Sparse Matrix

print("-----------------------SciPy Sparse Data------------------")
print("CSR: Compressed Sparse row ,CSC: Compressed Sparse column")
arr = np.array([0, 0, 0, 0, 0, 1, 1, 0, 2])
print(csr_matrix(arr))

# endregion

# region parse Matrix Methods

print("----------------Sparse Matrix Methods----------------")

print("   ")

print("Viewing stored data (non zero items) with the data property:")
arr = np.array([[0, 0, 0], [0, 0, 1], [1, 0, 2]])
print(csr_matrix(arr).data)

print("Counting non zeros with the count_nonzero() method:")
arr = np.array([[0, 0, 0], [0, 0, 1], [1, 0, 2]])
print(csr_matrix(arr).count_nonzero())

print("Removing zero-entries from the matrix with the eliminate_zeros() method:")
mat = csr_matrix(arr)
mat.eliminate_zeros()
print(mat)

print("Eliminating duplicate entries with the sum_duplicates() method:")
mat = csr_matrix(arr)
mat.sum_duplicates()
print(mat)

print("Converting from csr to csc with the tocsc() method:")
newarr = csr_matrix(arr).tocsc()
print(newarr)
# endregion

# region Scipy Graphs
print("------------------------Scipy Graphs------------------------")
print(
    "Adjacency matrix: Adjacency matrix is a nxn matrix where n is the number of elements in a graph.And the values represents the connection between the elements.")

print("---------Connected Components------------")
print("Find all of the connected components with the connected_components() method.")

arr = np.array([
    [0, 1, 2],
    [1, 0, 0],
    [2, 0, 0]
])

newarr = csr_matrix(arr)
print(connected_components(newarr))

print("-----------Dijkstra---------------")
print("Note:Use the dijkstra method to find the shortest path in a graph from one element to another.")
arr = np.array([
    [0, 1, 2],
    [1, 0, 0],
    [2, 0, 0]
])

newarr = csr_matrix(arr)

print(dijkstra(newarr, return_predecessors=True, indices=0))

print("-------------Floyd Warshall------------")
print("Note: Use the floyd_warshall() method to find shortest path between all pairs of elements.")
arr = np.array([
    [0, 1, 2],
    [1, 0, 0],
    [2, 0, 0]
])

newarr = csr_matrix(arr)
print(floyd_warshall(newarr, return_predecessors=True))

print("   ")
print("---------------Bellman Ford---------------")
print(
    "Note: The bellman_ford() method can also find the shortest path between all pairs of elements, but this method can handle negative weights as well.")
arr = np.array([
    [0, -1, 2],
    [1, 0, 0],
    [2, 0, 0]
])

newarr = csr_matrix(arr)
print(bellman_ford(newarr, return_predecessors=True, indices=0))

print("   ")

print("------------------Depth First Order------------")
print("The depth_first_order() method returns a depth first traversal from a node.")

print("This function takes following arguments:")
print("*the graph.")
print("*the starting element to traverse graph from.")
arr = np.array([
    [0, 1, 0, 1],
    [1, 1, 1, 1],
    [2, 1, 1, 0],
    [0, 1, 0, 1]
])

newarr = csr_matrix(arr)
print(depth_first_order(newarr, 1))

print("    ")
print("----------------------Breadth First Order--------------------")
print("The breadth_first_order() method returns a breadth first traversal from a node.")

print("This function takes following arguments:")
print("*the graph.")
print("*the starting element to traverse graph from.")
arr = np.array([
    [0, 1, 0, 1],
    [1, 1, 1, 1],
    [2, 1, 1, 0],
    [0, 1, 0, 1]
])

newarr = csr_matrix(arr)
print(breadth_first_order(newarr, 1))

print("    ")
# endregion

# region spacial data

print("---------------SciPy Spatial Data--------------")
print("--------Triangulation-------")

points = np.array([
    [2, 4],
    [3, 4],
    [3, 0],
    [2, 2],
    [4, 1]
])

simplices = Delaunay(points).simplices

plt.triplot(points[:, 0], points[:, 1], simplices)
plt.scatter(points[:, 0], points[:, 1], color='r')
plt.show()

print("   ")
print("------------Convex Hull------------")
points = np.array([
    [2, 4],
    [3, 4],
    [3, 0],
    [2, 2],
    [4, 1],
    [1, 2],
    [5, 0],
    [3, 1],
    [1, 2],
    [0, 2]
])

hull = ConvexHull(points)
hull_points = hull.simplices

plt.scatter(points[:, 0], points[:, 1])
for simplex in hull_points:
    plt.plot(points[simplex, 0], points[simplex, 1], 'k-')

plt.show()

print("-----------------KDTrees------------")
points = [(1, -1), (2, 3), (-2, 3), (2, -3)]
kdtree = KDTree(points)
res = kdtree.query((1, 1))
print(res)

print("    ")

print("--------------Distance Matrix-------------")
print("          ")

print("----------Euclidean distance--------------")
p1 = (1, 0)
p2 = (10, 2)

res = euclidean(p1, p2)
print(res)

print("   ")

print("-----------Cityblock Distance (Manhattan Distance)--------")

res = cityblock(p1, p2)
print(res)

print("-------------Cosine Distance--------------")

res = cosine(p1, p2)
print(res)

print("-------------Hamming Distance--------------")
p1 = (True, False, True)
p2 = (False, True, True)

res = hamming(p1, p2)
print(res)

# endregion

# region Matlab Arrays

arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, ])

# Export:
io.savemat('arr.mat', {"vec": arr})

# Import:
mydata = io.loadmat('arr.mat')

print(mydata)

print("To print only array: Use mydata[vec]")
print(mydata['vec'])

print("Note: *We can see that the array originally was 1D, but on extraction it has increased one dimension.")
print("      *In order to resolve this we can pass an additional argument squeeze_me=True:")
# Import:
mydata = io.loadmat('arr.mat', squeeze_me=True)

print(mydata['vec'])

# endregion

# region Scipy Interpolation

print("---------------1D Interpolation----------------")
xs = np.arange(10)
ys = 2 * xs + 1

interp_func = interp1d(xs, ys)
newarr = interp_func(np.arange(2.1, 3, 0.1))
print(newarr)

print("-----------------Spline Interpolation---------------------")
xs = np.arange(10)
ys = xs ** 2 + np.sin(xs) + 1

interp_func = UnivariateSpline(xs, ys)
newarr = interp_func(np.arange(2.1, 3, 0.1))
print(newarr)

print("--------------------Interpolation with Radial Basis Function-----------------")
xs = np.arange(10)
ys = xs ** 2 + np.sin(xs) + 1

interp_func = Rbf(xs, ys)
newarr = interp_func(np.arange(2.1, 3, 0.1))
print(newarr)

# endregion

# endregion
