# NumPy Learning by Code 🧠🐍

A **complete NumPy learning repository** designed to teach NumPy **from zero to advanced level** using a **step-by-step, topic-wise, code-first approach**.

This repository follows a **proper learning sequence**, so each concept builds naturally on the previous one.

---

## 📌 What is NumPy?

NumPy (Numerical Python) is the core library for:
- Numerical computing
- Array programming
- Data science
- Machine learning
- Scientific computing in Python

It provides **fast, memory-efficient, multi-dimensional arrays** and powerful mathematical operations.

---

## 🧭 Complete NumPy Learning Roadmap (In Sequence)

---

## 1️⃣ NumPy Basics & Setup
- What is NumPy?
- Why NumPy is faster than Python lists
- Installing NumPy
- Importing NumPy
- NumPy naming conventions
- Understanding `ndarray`
- NumPy vs List (performance basics)

---

## 2️⃣ NumPy Arrays Fundamentals
- Creating NumPy arrays
- 1D arrays
- 2D arrays (matrices)
- 3D & n-dimensional arrays
- Array shape & structure
- `ndim`, `shape`, `size`
- `dtype`
- `itemsize`
- `nbytes`

---

## 3️⃣ Array Creation Methods
- `array()`
- `asarray()`
- `zeros()`
- `ones()`
- `empty()`
- `full()`
- `arange()`
- `linspace()`
- `logspace()`
- `eye()` / `identity()`
- Creating arrays from Python lists & tuples

---

## 4️⃣ Data Types (dtype) in NumPy
- What is `dtype`
- Integer types
- Float types
- Boolean type
- String type
- Complex numbers
- Type conversion (`astype`)
- Custom dtypes
- Type casting rules

---

## 5️⃣ Indexing & Slicing
- Basic indexing
- Negative indexing
- Slicing arrays
- Slicing 2D arrays
- Step slicing
- Row & column selection
- Modifying values using indexing
- Views vs copies (intro)

---

## 6️⃣ Advanced Indexing
- Fancy indexing
- Indexing using lists
- Indexing using arrays
- Boolean indexing
- Conditional selection
- Filtering arrays using conditions
- Replacing values conditionally

---

## 7️⃣ Shape Manipulation
- `reshape()`
- `flatten()`
- `ravel()`
- `transpose()`
- `T`
- `swapaxes()`
- `expand_dims()`
- `squeeze()`
- Understanding shape compatibility

---

## 8️⃣ Copy vs View (Very Important)
- What is a view
- What is a copy
- How slicing creates views
- When NumPy creates copies
- `copy()` method
- Memory behavior
- Common mistakes

---

## 9️⃣ Joining & Splitting Arrays
### Joining
- `concatenate()`
- `vstack()`
- `hstack()`
- `dstack()`
- `column_stack()`
- `row_stack()`

### Splitting
- `split()`
- `array_split()`
- `vsplit()`
- `hsplit()`

---

## 🔟 Mathematical Operations
- Element-wise operations
- Addition, subtraction, multiplication
- Division & floor division
- Modulus
- Power
- Unary operations
- `abs()`
- `round()`
- `floor()` & `ceil()`

---

## 1️⃣1️⃣ Universal Functions (ufuncs)
- What are ufuncs
- Trigonometric functions
- Exponential functions
- Logarithmic functions
- Comparison functions
- Bitwise functions
- Performance benefits of ufuncs

---

## 1️⃣2️⃣ Broadcasting (Core Concept)
- What is broadcasting
- Broadcasting rules
- Scalar broadcasting
- Vector broadcasting
- Matrix broadcasting
- Common broadcasting errors
- Real-world use cases

---

## 1️⃣3️⃣ Aggregation & Statistical Functions
- `sum()`
- `min()` & `max()`
- `mean()`
- `median()`
- `std()`
- `var()`
- `argmin()` & `argmax()`
- Axis-based calculations
- Cumulative operations

---

## 1️⃣4️⃣ Sorting & Searching
- `sort()`
- `argsort()`
- Sorting along axis
- `where()`
- `searchsorted()`
- Conditional searching
- Unique values (`unique()`)

---

## 1️⃣5️⃣ Random Module
- Random numbers basics
- `rand()`
- `randn()`
- `randint()`
- `choice()`
- `shuffle()`
- `permutation()`
- Setting random seed
- Probability distributions

---

## 1️⃣6️⃣ Linear Algebra
- Dot product
- Matrix multiplication
- `matmul()`
- Determinant
- Inverse matrix
- Eigenvalues & eigenvectors
- Solving linear equations
- Identity matrices

---

## 1️⃣7️⃣ Logical & Comparison Operations
- Logical operators
- `any()` & `all()`
- Comparison operators
- Masking arrays
- Combining conditions
- Filtering real datasets

---

## 1️⃣8️⃣ Handling Missing & Special Values
- `NaN`
- `Inf` & `-Inf`
- Checking missing values
- Replacing NaN values
- Ignoring NaNs in calculations
- Cleaning numerical data

---

## 1️⃣9️⃣ Performance Optimization
- Vectorization
- Avoiding loops
- Memory efficiency
- Using correct dtypes
- NumPy vs pure Python performance
- Best practices

---

## 2️⃣0️⃣ NumPy with Real-World Examples
- Data preprocessing
- Feature scaling
- Normalization
- Standardization
- Image array basics
- Time-series numeric data
- ML-ready numeric data

---

## 🧠 Learning Philosophy

- Learn concepts **in order**
- Practice with **small code snippets**
- Understand **why**, not just how
- Write clean, readable NumPy code
- Focus on **real-world & interview usage**

---

## 🎯 Who Should Use This Repo?
- Python beginners
- Data Science learners
- ML / AI aspirants
- Interview preparation
- Anyone who wants **strong NumPy fundamentals**

---

## 🚀 Next Step

Each topic will be added with:
- Explanation
- Code snippets
- Outputs
- Important notes
- Common mistakes

---

## ⭐ Support

If this repository helps you:
- Star ⭐ the repo
- Share with learners
- Practice consistently

Happy NumPy Learning 🚀

📌 1. NumPy Basics & Setup
```python
# Requirements (Before You Start)
🔹 Step 0: Prerequisites

=> You should already know:
=> Basic Python syntax
=> What are variables
=> What is a list
=> How to run a Python file or use terminal / VS Code / Jupyter

If you know how to write:
a = [1, 2, 3]
print(a)

You’re good to go 👍

🔹 Step 1: What is NumPy?
Requirement:
Basic understanding of Python data structures

What is NumPy?
=> NumPy (Numerical Python) is a Python library used for:
=> Fast numerical calculations
=> Working with arrays and matrices
=> Scientific computing
=> Data analysis, AI, ML, and Deep Learning

Why NumPy exists?
Python lists are:
=> Slow for large data
=> Not optimized for math operations

NumPy solves this by:
=> Using C language internally
=> Storing data in continuous memory
=> Supporting vectorized operations

🔹 Step 2: Why NumPy is Faster than Python Lists
Requirement:
=> Understanding of Python lists

| Python List                 | NumPy Array                |
| --------------------------- | -------------------------- |
| Stores different data types | Stores same data type      |
| Slow loops                  | Fast vectorized operations |
| High memory usage           | Low memory usage           |
| Python-level execution      | C-level execution          |

Example (Conceptual):
Python list addition:
result = []
for i in range(len(a)):
    result.append(a[i] + b[i])


NumPy:
result = a + b

Note: No loop, no complexity, super fast

🔹 Step 3: Installing NumPy
Requirement:
=> Python installed on your system
=> Internet connection

# Step-by-step Installation
# Step 3.1: Check Python version
python --version
or
python3 --version
Note: Python 3.7 or above recommended

# Step 3.2: Install NumPy
=> pip install numpy

# Step 3.3: Verify installation
python

import numpy
print(numpy.__version__)

🔹 Step 4: Importing NumPy
Requirement:

=> NumPy installed successfully
Standard Import (BEST PRACTICE)
import numpy as np

Why np?
=> Community standard
=> Short and readable
=> Used in almost all projects

❌ Avoid:
=> import numpy

✔ Prefer:
=> import numpy as np


🔹 Step 5: NumPy Naming Conventions
# Requirement:
=> Basic Python import knowledge

Standard Conventions:
| Item      | Convention |
| --------- | ---------- |
| Library   | `numpy`    |
| Alias     | `np`       |
| Array     | `arr`      |
| Matrix    | `mat`      |
| Shape     | `shape`    |
| Data type | `dtype`    |


Example:
import numpy as np

arr = np.array([1, 2, 3])
print(arr)

```
📌 2. NumPy Arrays Fundamentals (Deep & Simple Guide)
```python
🔹 Step 1: Creating NumPy Arrays
Requirement:
=> Python list knowledge

Main Method: np.array()
This function converts Python lists (or tuples) into NumPy arrays.

import numpy as np

# Creating a NumPy array from a Python list
arr = np.array([1, 2, 3, 4, 5])

print(arr)
print(type(arr))  # Shows that this is a NumPy ndarray


What’s happening:
=> np.array() takes a list
=> Converts it into a NumPy ndarray
=> All elements become the same data type

🔹 Step 2: 1D Arrays (One-Dimensional)

Requirement:
=> Understanding of linear lists

What is a 1D array?
=> Looks like a normal list
=> Has only one direction

import numpy as np

arr_1d = np.array([10, 20, 30, 40])

print(arr_1d)
print(arr_1d.ndim)   # Number of dimensions
print(arr_1d.shape)  # Size of array  (column, row)

🔹 Step 3: 2D Arrays (Matrices)
Requirement:
=> Nested lists concept

What is a 2D array?
=> Rows and columns
=> Used in matrices, tables, images

import numpy as np

arr_2d = np.array([
    [1, 2, 3],
    [4, 5, 6]
])
print(arr_2d.ndim)   # 2 dimensions
print(arr_2d.shape)  # (rows, columns)

🔹 Step 4: 3D & N-Dimensional Arrays
Requirement:
=> Understanding of nested structures

What is a 3D array?
=> Multiple 2D matrices stacked together
=> Used in videos, deep learning, tensors

import numpy as np
arr_3d = np.array([
    [[1, 2], [3, 4]],
    [[5, 6], [7, 8]]
])

print(arr_3d)

print(arr_3d.ndim)
print(arr_3d.shape)

🔹 Step 5: Array Shape & Structure
Requirement:
=> Understanding of rows and columns

# shape
=> # Tells how the data is structured

import numpy as np
arr = np.array([
    [10, 20, 30],
    [40, 50, 60]
])

print(arr.shape)  # (2, 3) => (2 rows, 3 columns)

🔹 Step 6: ndim (Number of Dimensions)

import numpy as np
arr = np.array([1, 2, 3])
print(arr.ndim)  # 1

🔹 Step 7: size (Total Elements)

import numpy as np
arr = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

print(arr.size) # 6

Explanation:
=> Counts all elements
=> Rows × Columns = Total

🔹 Step 8: dtype (Data Type)
Requirement:
=> Basic data type knowledge

import numpy as np
arr = np.array([1, 2, 3])
print(arr.dtype)   #int64   # (or int32 depending on system)

# Mixed data example:
import numpy as np
arr = np.array([1, 2.5, 3])
print(arr.dtype)  # float64

Why?
=> NumPy converts everything to one common type
=> This makes operations fast

🔹 Step 9: itemsize (Memory of One Element)
Requirement:
Understanding of bytes

import numpy as np
arr = np.array([1, 2, 3])
print(arr.itemsize)   # 8

Meaning:
Each element uses 8 bytes in memory

🔹 Step 10: nbytes (Total Memory Used)
Requirement:
=> size and itemsize knowledge

import numpy as np
arr = np.array([1, 2, 3, 4])
print(arr.nbytes)   # 32

note:
How it’s calculated:
nbytes = size × itemsize

🔹 Step 11: All Properties Together (Real Example)
import numpy as np

arr = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

print("Dimensions:", arr.ndim)
print("Shape:", arr.shape)
print("Total elements:", arr.size)
print("Data type:", arr.dtype)
print("Bytes per element:", arr.itemsize)
print("Total memory:", arr.nbytes)

# Quick Reference Table

| Property   | Meaning              |
| ---------- | -------------------- |
| `ndim`     | Number of dimensions |
| `shape`    | Structure of array   |
| `size`     | Total elements       |
| `dtype`    | Data type            |
| `itemsize` | Memory per element   |
| `nbytes`   | Total memory used    |

```
📌 3. NumPy Array Creation Methods (Complete Guide)
```python
✅ Requirements
# Before starting, you should know:
How to import NumPy

import numpy as np
=> What is an ndarray
=> Basic Python lists & tuples

🔹 1. np.array() – Create Array from List or Tuple

Purpose:
Convert Python list or tuple into a NumPy array

Example (List):

import numpy as np
arr = np.array([1, 2, 3, 4])
print(arr)
print(type(arr))  # Shows it's a NumPy array

Example (Tuple):

arr = np.array((10, 20, 30))
print(arr)

Important Points:
=> Automatically detects data type
=> Converts all elements to same type

🔹 2. np.asarray() – Convert to Array (Without Copy)
Purpose:
=> Converts input to NumPy array
=> Does NOT create a copy if already an array

Example:

import numpy as np
lst = [1, 2, 3]

# np.array() converts the Python list into a NEW NumPy array
# It always creates a copy of the data
arr1 = np.array(lst)

# np.asarray() also converts the list into a NumPy array
# But it avoids copying data if the input is already a NumPy array
# (here input is a list, so it still creates a new array)
arr2 = np.asarray(lst)

print(arr1)
print(arr2)

Why important?
=> Faster
=> Memory efficient

Check behavior:

import numpy as np
arr = np.array([1, 2, 3])
arr_as = np.asarray(arr)

print(arr is arr_as)  # True (same memory)


🔹 3. np.zeros() – Array Filled with Zeros
Purpose:
=> Create array with all values = 0

# 1D Example:

import numpy as np
arr = np.zeros(5)
print(arr)

# 2D Example:

import numpy as np
arr = np.zeros((2, 3))
print(arr)

Use Case:
=> Initializing weights
=> Placeholder arrays

🔹 4. np.ones() – Array Filled with Ones
Purpose:
=> Create array with all values = 1

import numpy as np
arr = np.ones((3, 4))
print(arr)

🔹 5. np.empty() – Create Array Without Initialization
Purpose:
=> Creates array but does NOT set values

import numpy as np
arr = np.empty((2, 3))
print(arr)

Important:
=> Contains random garbage values
=> Faster than zeros() and ones()
# Note: Use only when you plan to fill it immediately.


🔹 6. np.full() – Array with Custom Value
Purpose:
=> Fill array with a specific value

import numpy as np
arr = np.full((3, 3), 7)
print(arr)

🔹 7. np.arange() – Range of Values (Like range())
Purpose:
=> Create array with evenly spaced values

import numpy as np
arr = np.arange(0, 10)
print(arr)

# With Step:

import numpy as np
arr = np.arange(0, 10, 2)
print(arr)

# Note: Important:--> End value is excluded

🔹 8. np.linspace() – Fixed Number of Values
Purpose:
=> Create N numbers between two values (inclusive)

import numpy as np  
# np.linspace() is used to create numbers evenly spaced between two values
# Syntax: np.linspace(start, end, total_numbers)

# Create 5 numbers between 0 and 10 (both 0 and 10 are included)
arr = np.linspace(0, 10, 5)
print(arr)


# Difference from arange():
| arange       | linspace     |
| ------------ | ------------ |
| Step-based   | Count-based  |
| End excluded | End included |

🔹 9. np.logspace() – Logarithmic Scale
Purpose:
=> Values spaced evenly on a log scale

import numpy as np 
# np.logspace() creates numbers that are evenly spaced on a LOG scale
# Syntax: np.logspace(start_power, end_power, total_numbers)

# Create 3 numbers between 10¹ and 10³ (on a logarithmic scale)
arr = np.logspace(1, 3, 3)

# Print the generated NumPy array
print(arr)


Explanation:
=> 10¹ → 10² → 10³

Use Case:
=> Scientific calculations
=> Machine learning scales

🔹 10. np.eye() – Identity Matrix
Purpose:
=> Diagonal elements = 1
=> Others = 0


import numpy as np 
# np.eye() creates an IDENTITY MATRIX
# Identity matrix means:
# - All diagonal values are 1
# - All other values are 0

# Create a 4 x 4 identity matrix
arr = np.eye(4)

# Create an identity-like matrix with 3 rows and 5 columns
arr = np.eye(3, 5)


# Print the identity matrix
print(arr)


🔹 11. np.identity() – Square Identity Matrix
Purpose:
=> Same as eye() but only square matrix

import numpy as np 
arr = np.identity(3)
print(arr)

np.identity(4)  # Only square matrix
np.eye(4)       # Square or rectangular

🔹 12. Creating Arrays from Python Lists & Tuples
# 2D List:

import numpy as np 
lst = [
    [1, 2, 3],
    [4, 5, 6]
]

arr = np.array(lst)
print(arr)

# Tuple:

import numpy as np 
tup = (10, 20, 30)
arr = np.array(tup)
print(arr)

🔹 13. Additional Important Creation Methods (Often Missed)

=> # np.zeros_like()

import numpy as np 
# Base NumPy array (this will be used as a reference)
base = np.array([1, 2, 3])

# np.zeros_like() creates a new array
# - Same shape as 'base'
# - All values are set to 0
arr = np.zeros_like(base)

print(arr)  # Output: [0 0 0]

=> # np.ones_like()

# - All values are set to 1
arr = np.ones_like(base)

=> # np.full_like()

arr = np.full_like(base, 9)
print(arr)  # Output: [9 9 9]

# Quick Summary Table

| Method       | Purpose                |
| ------------ | ---------------------- |
| `array()`    | Create from list/tuple |
| `asarray()`  | Convert without copy   |
| `zeros()`    | All zeros              |
| `ones()`     | All ones               |
| `empty()`    | Uninitialized          |
| `full()`     | Custom value           |
| `arange()`   | Range with step        |
| `linspace()` | Fixed count            |
| `logspace()` | Log scale              |
| `eye()`      | Identity matrix        |
| `identity()` | Square identity        |
| `*_like()`   | Copy shape             |

```
📌 4. Data Types (dtype) in NumPy
```python
✅ Requirements
Before starting, you should know:
=> How to create NumPy arrays
=> What is an ndarray
=> Basic Python data types (int, float, bool, str)

🔹 1. What is dtype?
Meaning:
dtype (data type) tells:

=> What kind of data is stored in the array
=> How much memory each element uses
=> How NumPy treats values during operations

Example:

import numpy as np
arr = np.array([1, 2, 3])
print(arr.dtype)  # Shows data type of array elements

🔹 2. Integer Types in NumPy
Purpose:
=> Store whole numbers

# Common Integer Types:

| Type    | Description    |
| ------- | -------------- |
| `int8`  | 8-bit integer  |
| `int16` | 16-bit integer |
| `int32` | 32-bit integer |
| `int64` | 64-bit integer |


Example:
import numpy as np

# Create a NumPy array with explicit data type (int32)
# dtype=np.int32 means:
# - Each number is stored as a 32-bit integer
# - Uses less memory than default int64 on most systems
arr = np.array([10, 20, 30], dtype=np.int32)

# Print the array values
print(arr)   # Output: [10 20 30]

# Print the data type of elements stored in the array
print(arr.dtype)  # Output: int32

# itemsize tells how much memory ONE element uses
# The value is in bytes
print(arr.itemsize)  # Output: 4


# Memory comparison example
np.array([1, 2, 3], dtype=np.int32).itemsize  # 4 bytes
np.array([1, 2, 3], dtype=np.int64).itemsize  # 8 bytes

# Note: 
Explanation:

=> Smaller integer → less memory
=> Use wisely for large datasets

🔹 3. Float Types in NumPy
Purpose:
=> Store decimal numbers

Common Float Types:
| Type      | Description              |
| --------- | ------------------------ |
| `float16` | Low precision            |
| `float32` | Medium precision         |
| `float64` | High precision (default) |


Example:

import numpy as np
arr = np.array([1.5, 2.8, 3.1], dtype=np.float32)
print(arr)
print(arr.dtype)

# Automatic Conversion:

import numpy as np
arr = np.array([1, 2.5, 3])
print(arr.dtype)  # Converted to float automatically

🔹 4. Boolean Type
Purpose:
=> Store True or False

Example:

import numpy as np
arr = np.array([True, False, True])
print(arr)
print(arr.dtype)

# Numeric Behavior:

import numpy as np
arr = np.array([True, False, True])
print(arr + 1)

# Note: Explanation:--->
=> True = 1
=> False = 0

🔹 5. String Type
Purpose:
=> Store text data

import numpy as np 

# Create a NumPy array containing text (strings)
arr = np.array(["apple", "banana", "mango"])

# Print the array values
print(arr)   # Output: ['apple' 'banana' 'mango']

# Print the data type of the array elements
print(arr.dtype)  # Output: <U6

# Note:
NumPy automatically:
=> Checks all strings
=> Picks the longest length
=> Sets dtype to <U6

🔹 6. Complex Numbers
Purpose:
=> Store numbers with real + imaginary parts

Example:

import numpy as np 
arr = np.array([1+2j, 3+4j])
print(arr)
print(arr.dtype)

# Note:
=> # Access Parts:

print(arr.real)  # real part
print(arr.imag)  # imaginary part

🔹 7. Type Conversion Using astype()
Purpose:
=> Convert array from one dtype to another

Example:

import numpy as np 
arr = np.array([1.2, 2.5, 3.9])
new_arr = arr.astype(int)

print(new_arr)
print(new_arr.dtype)

# Note: Decimal part is removed, not rounded

# String to Integer:

import numpy as np 
arr = np.array(["1", "2", "3"])
new_arr = arr.astype(int)
print(new_arr)


🔹 8. Custom dtypes (Structured Arrays)
Purpose:
=> Store different data types in one array
=> (Like a table or record)

Example:

import numpy as np 

# Define a CUSTOM data type (structured dtype)
# Each element will behave like a small record (row)
dtype = [
    ("name", "U10"),   # 'name' → text (Unicode), max 10 characters
    ("age", "i4"),     # 'age' → integer, uses 4 bytes
    ("salary", "f8")   # 'salary' → decimal number, uses 8 bytes
]

# Create a NumPy array using the custom structure
# Each row must follow the order: (name, age, salary)
arr = np.array(
    [
        ("Monu", 25, 50000.0),  # First person record
        ("Amit", 30, 60000.0)   # Second person record
    ],
    dtype=dtype  # Tell NumPy to use the structured dtype
)

# Print the full structured array
print(arr)

# Print detailed information about the structure
print(arr.dtype)

# Easy way to access data
print(arr["name"])    # ['Monu' 'Amit']
print(arr["age"])     # [25 30]
print(arr["salary"])  # [50000. 60000.]

🔹 9. Type Casting Rules in NumPy
=> # Rule 1: Safe Casting (Automatic)

import numpy as np 
arr = np.array([1, 2, 3.5])
print(arr.dtype)  # int → float

=> # Rule 2: Unsafe Casting (Manual)

import numpy as np 
arr = np.array([1.9, 2.1])
print(arr.astype(int))  # data loss

=> # Rule 3: Boolean Casting

import numpy as np 
arr = np.array([0, 1, 2, 3])
print(arr.astype(bool))

Explanation:
=> 0 → False
=> Non-zero → True

🔹 10. Checking All dtype Information Together

import numpy as np 
arr = np.array([1, 2, 3], dtype=np.int16)

print("Array:", arr)
print("Data type:", arr.dtype)
print("Bytes per element:", arr.itemsize)
print("Total memory:", arr.nbytes)


Quick Summary Table
| Type             | Description      |
| ---------------- | ---------------- |
| `int`            | Whole numbers    |
| `float`          | Decimal numbers  |
| `bool`           | True/False       |
| `str`            | Text             |
| `complex`        | Real + Imaginary |
| `astype()`       | Type conversion  |
| Structured dtype | Mixed data       |
```
📌 5. Indexing & Slicing
```python
✅ Requirements
Before starting, you should know:
=> How to create NumPy arrays
=> What is a 1D and 2D array
=> Basic Python indexing (list[0])

🔹 1. Basic Indexing (1D Array)
Purpose:
=> # Access a single element using its position

import numpy as np
arr = np.array([10, 20, 30, 40, 50])

print(arr[0])   # First element
print(arr[2])   # Third element
print(arr[4])   # Fifth element

# Important:
=> # Indexing starts from 0

🔹 2. Negative Indexing
Purpose:
=> Access elements from the end

print(arr[-1])  # Last element
print(arr[-2])  # Second last element

Explanation:
=> # -1 always means last element

🔹 3. Slicing 1D Arrays
Purpose:
=> # Extract multiple values at once

# Syntax:
array[start : stop : step]


print(arr[1:4])   # From index 1 to 3
print(arr[:3])    # From start to index 2
print(arr[2:])    # From index 2 to end

Explanation:
=> # Stop index is not included

🔹 4. Step Slicing
Purpose:
=> # Skip elements

import numpy as np 

# Create a NumPy array
arr = np.array([10, 20, 30, 40, 50, 60])

# arr[start : stop : step]

# Take every 2nd element starting from index 0

Index:  0   1   2   3   4   5
Value: 10  20  30  40  50  60
Take:  ✔       ✔       ✔

print(arr[::2])    # Output: [10 30 50]

# Take every 2nd element starting from index 1

Index:  0   1   2   3   4   5
Value: 10  20  30  40  50  60
Take:      ✔       ✔       ✔

print(arr[1::2])   # Output: [20 40 60]


🔹 5. Slicing 2D Arrays
Requirement:
=> # Understanding of rows and columns


import numpy as np 
arr_2d = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

print(arr_2d)

import numpy as np  # Import NumPy library

# Create a 2D NumPy array (matrix)
# Think of this like a table with rows and columns
arr_2d = np.array([
    [10, 20, 30],   # Row 0
    [40, 50, 60],   # Row 1
    [70, 80, 90]    # Row 2
])

# Access element at row 0 and column 0
# Row index = 0 (first row)
# Column index = 0 (first column)
print(arr_2d[0, 0])   # Output: 10

# Access element at row 1 and column 2
# Row index = 1 (second row)
# Column index = 2 (third column)
print(arr_2d[1, 2])   # Output: 60

# How 2D indexing works (very simple)
# Visual representation (easy to understand)

        Col 0   Col 1   Col 2
       ---------------------
Row 0 |  10      20      30
Row 1 |  40      50      60
Row 2 |  70      80      90

#  Example 1: arr_2d[0, 0]
 => # Go to row 0 (first row)
 => # Then go to column 0 (first column)

# output: 10

🔹 7. Row Selection

import numpy as np  

# Create a 2D array (think of it as a table)
arr_2d = np.array([
    [10, 20, 30],   # Row 0
    [40, 50, 60],   # Row 1
    [70, 80, 90]    # Row 2
])

=> # Select a FULL row

# Select the entire row at index 1 (second row)
print(arr_2d[1])   # Output: [40 50 60]

=> # Select MULTIPLE rows (row slicing)

# Select rows from index 0 up to (but not including) index 2
print(arr_2d[0:2])

# Output of arr_2d[0:2]
[[10 20 30]
 [40 50 60]]

# More simple examples
arr_2d[:2]   # First two rows (row 0 and 1)
arr_2d[1:]   # From row 1 till end
arr_2d[-1]   # Last row

🔹 8. Column Selection

import numpy as np  

# Create a 2D NumPy array (table)
arr_2d = np.array([
    [10, 20, 30],   # Row 0
    [40, 50, 60],   # Row 1
    [70, 80, 90]    # Row 2
])

# Select a FULL column

# Select ALL rows, but only column at index 1 (second column)
print(arr_2d[:, 1])

# What is happening (simple meaning)
=> : → take all rows
=> 1 → take column index 1
=> Result → one full column

# Output
=> # [20 50 80]

#  Select MULTIPLE columns

# Select ALL rows, and columns from index 0 up to (but not including) 2
print(arr_2d[:, 0:2])

What is happening
=> : → all rows
=> 0:2 → column 0 and column 1
=> Column 2 is NOT included

Visual understanding

            Col 0   Col 1   Col 2
          -----------------------
Row 0   →     ✔      ✔      ✖
Row 1   →     ✔      ✔      ✖
Row 2   →     ✔      ✔      ✖

Easy memory trick
=> : before comma → all rows
=> : after comma → all columns

# One-line rule to remember
=> arr_2d[:, col] → one column
=> arr_2d[:, start:end] → many columns

# More simple examples
arr_2d[:, -1]   # Last column
arr_2d[:, :1]   # First column (keeps 2D shape)

🔹 9. Slicing Rows and Columns Together

import numpy as np

# Create a 2D NumPy array (table)
arr_2d = np.array([
    [10, 20, 30],   # Row 0
    [40, 50, 60],   # Row 1
    [70, 80, 90]    # Row 2
])

# Select:
# Rows from index 0 up to (but not including) 2
# Columns from index 1 up to (but not including) 3
print(arr_2d[0:2, 1:3])

=> # What is happening (very simple)
General format:

# arr_2d[row_start:row_end, col_start:col_end]

=> First part → rows
=> Second part → columns
=> Start index → included
=> End index → NOT included

# Step-by-step breakdown

0:2 → rows

=> Row 0 → ✔ included
=> Row 1 → ✔ included
=> Row 2 → ❌ excluded

1:3 → columns
=> Column 1 → ✔ included
=> Column 2 → ✔ included
=> Column 3 → ❌ excluded (does not exist anyway)


# Easy memory trick
=> Before comma → rows
=> After comma → columns

#  One-line rule to remember

=> arr_2d[a:b, c:d] → take rows a to b-1 and columns c to d-1
=> arr_2d[1:3, 0:2]  # Rows 1–2, Columns 0–1
=> arr_2d[:2, :2]    # Top-left corner
=> arr_2d[:, 1:]     # All rows, columns from 1 to end

# More simple examples
arr_2d[1:3, 0:2]  # Rows 1–2, Columns 0–1
arr_2d[:2, :2]    # Top-left corner
arr_2d[:, 1:]     # All rows, columns from 1 to end


🔹 10. Modifying Values Using Indexing

=> # Modify a single value (1D array)

import numpy as np 

# Create a 1D NumPy array
arr = np.array([10, 20, 30, 40, 50])

# Change the value at index 0 (first element) to 100
arr[0] = 100

# Print the updated array
print(arr)

=> # Modify MULTIPLE values (slice assignment)

# Replace values from index 1 up to (but not including) index 4 with 999
arr[1:4] = 999

# Print the updated array
print(arr)

What is happening
=> # 1:4 → index 1, 2, 3
=> # All selected elements get same new value
=> # NumPy changes them in one operation

=> # Modify values in a 2D array (rows & columns)

import numpy as np 

# Create a 2D NumPy array (table with rows & columns)
arr_2d = np.array([
    [10, 20, 30],   # Row 0
    [40, 50, 60],   # Row 1
    [70, 80, 90]    # Row 2
])

# arr_2d[0, :] means:
# 0  → select FIRST row (top row)
# :  → select ALL columns in that row
# = 0 → replace every selected value with 0
arr_2d[0, :] = 0

# Print updated array
print(arr_2d)

# Output

[[ 0  0  0]
 [40 50 60]
 [70 80 90]]

# What exactly happened (simple words)
=> # NumPy selected the first row
=> # It selected all columns inside that row
=> # Then it replaced every value with 0

# Visual understanding

Before

Row 0 → [10 20 30]
Row 1 → [40 50 60]
Row 2 → [70 80 90]

After

Row 0 → [ 0  0  0]
Row 1 → [40 50 60]
Row 2 → [70 80 90]

# Easy rules to remember

import numpy as np

arr = np.array([
    [10, 20, 30],   # Row 0
    [40, 50, 60],   # Row 1
    [70, 80, 90]    # Row 2
])

# Visual view:

            Col 0   Col 1   Col 2
          -----------------------
Row 0   →     10     20     30
Row 1   →     40     50     60
Row 2   →     70     80     90

# Change ONE specific cell (row + column)

arr[1, 2] = 999

# Meaning:
=> # Row 1 (second row)
=> # Column 2 (third column)
=> # Changes only 60 → 999

# Change MANY cells in SAME ROW (some columns)

arr[1, 0:2] = 0

# Meaning:
=> Row 1
=> Columns 0 and 1
=> [40 50] → [0 0]

# Change FULL ROW

arr[0, :] = 5

# Meaning:
=> # Row 0
=> # All columns (:)
=> # [10 20 30] → [5 5 5]

# Change FULL COLUMN

arr[:, 1] = 777

# Meaning:
=> # All rows (:)
=> # Column 1
[20 50 80] → [777 777 777]


# Change MULTIPLE ROWS & MULTIPLE COLUMNS

arr[0:2, 1:3] = 99

Meaning:
=># Rows 0 and 1
=># Columns 1 and 2

# Select (not change) specific data

=> # Get one value
value = arr[2, 0]   # Row 2, Col 0 → 70

=> # Get one full row
row = arr[1, :]

=> # Get one full column

col = arr[:, 2]

# Mental shortcut (VERY IMPORTANT)
arr[ ROWS , COLUMNS ]

=> # Before comma → vertical movement (down)
=> # After comma → horizontal movement (right)

🔹 11. Boolean Indexing (IMPORTANT)
Purpose:
=> # Select elements based on condition

arr = np.array([10, 20, 30, 40, 50])
print(arr[arr > 25])  # Elements greater than 25

=> # Modify using condition:

arr[arr > 30] = 999  # All elements will be replace by 999
print(arr)


🔹 12. Fancy Indexing (Index with List)
Purpose:
=> # Select specific positions

import numpy as np  # Import NumPy library

# Create a NumPy array
arr = np.array([10, 20, 30, 40, 50])

# Use a LIST of indexes to pick specific positions
# [0, 2, 4] means:
# - pick index 0 → 10
# - pick index 2 → 30
# - pick index 4 → 50
print(arr[[0, 2, 4]]) # [10 30 50]

# What is happening (very simple)
=> # Normal indexing → picks one position
=> # Fancy indexing → picks many specific positions at once

🔹 Fancy indexing in 2D (quick look)

arr_2d = np.array([
    [10, 20, 30],
    [40, 50, 60],
    [70, 80, 90]
])

# Pick specific rows
print(arr_2d[[0, 2]])

Output:

[[10 20 30]
 [70 80 90]]

# Easy rule to remember
=> # Fancy indexing = Indexing with a list or array

🔹 13. Views vs Copies (VERY IMPORTANT)
View (Changes affect original array)

import numpy as np  # Import NumPy

# Original NumPy array
arr = np.array([1, 2, 3, 4, 5])

# Take a slice from index 1 to 3 (NOT a new array, it's a VIEW)
slice_arr = arr[1:4]

# Change all values inside the slice
slice_arr[:] = 99

# Print original array
print(arr)   # [ 1 99 99 99  5 ]

# What is happening (very brief)
=> # arr[1:4] → creates a VIEW
=> # View = shares same memory
=> # Changing slice → original array also changes

# Visual idea

arr = [1, 2, 3, 4, 5]
          ↑  ↑  ↑
        slice part

# Output: arr = [1, 99, 99, 99, 5]

# Copy (Changes do NOT affect original)

arr = np.array([1, 2, 3, 4, 5])

copy_arr = arr[1:4].copy()  # Explicit copy
copy_arr[:] = 88

print(arr)       # Original unchanged
print(copy_arr)

🔹 14. Checking If It’s a View or Copy

print(slice_arr.base)  # If not None → it's a view
print(copy_arr.base)   # None → it's a copy

# Quick Cheat Sheet

| Feature           | Example             |
| ----------------- | ------------------- |
| Basic indexing    | `arr[0]`            |
| Negative indexing | `arr[-1]`           |
| Slice             | `arr[1:4]`          |
| Step              | `arr[::2]`          |
| Row select        | `arr[1]`            |
| Column select     | `arr[:, 1]`         |
| Modify            | `arr[arr > 10] = 0` |
| Copy              | `arr.copy()`        |


```
📌 6. Advanced Indexing in NumPy
```python
✅ Requirements
# Before starting, you should know:
=> # Basic indexing & slicing
=> # What is a NumPy array
=> # Boolean values (True / False)

🔹 Step 1: Fancy Indexing (Core Concept)
# What is Fancy Indexing?
=> # Selecting elements using another list or array of indexes
=> # Unlike slicing, fancy indexing creates a copy

🔹 Step 2: Indexing Using Python Lists

# 1D Array Example


import numpy as np 

# Create a 1D NumPy array
arr = np.array([10, 20, 30, 40, 50])

# Fancy indexing:
# We pass a LIST of index positions [0, 2, 4]
# NumPy picks values from these positions
selected = arr[[0, 2, 4]]

# Print the new array
print(selected)  # [10, 30, 50]

# This part is the key
arr[[0, 2, 4]]

=> # [0, 2, 4] → tells NumPy which positions you want
=> # NumPy goes one by one:
=> # index 0 → 10
=> # index 2 → 30
=> # index 4 → 50

# Note: Very important thing to remember
=> # Fancy indexing always creates a COPY
=> # Original array is NOT affected
=> # Safe to modify

selected[0] = 999
print(arr)       # original stays same
print(selected)  # only selected changes

# Fancy Indexing vs Slicing

| Feature              | Fancy Indexing | Slicing  |
| -------------------- | -------------- | -------- |
| Uses list of indexes | ✅              | ❌        |
| Returns copy         | ✅              | ❌ (view) |
| Original changes     | ❌              | ✅        |


# 2D Array Example (Row Selection)

import numpy as np  # Import NumPy

# Create a 2D NumPy array (3 rows × 3 columns)
arr_2d = np.array([
    [10, 20, 30],   # Row index 0
    [40, 50, 60],   # Row index 1
    [70, 80, 90]    # Row index 2
])

# Fancy indexing for rows:
# [0, 2] means → pick row 0 and row 2
rows = arr_2d[[0, 2]]

# Print selected rows
print(rows)

# Output:-
[[10 20 30]
 [70 80 90]]

# Note: Important behavior
=> # Fancy indexing on rows returns a NEW array (COPY)
=> # Original arr_2d is not changed
=> # Safe to modify rows

rows[0, 0] = 999
print(arr_2d)  # original remains same

🔹 Step 3: Indexing Using NumPy Arrays
Purpose:

=> # Same as list indexing but faster and more flexible

import numpy as np 

# Create an array that stores index positions we want
indexes = np.array([1, 3])   # Means: pick position 1 and position 3

# Create a NumPy array with values
arr = np.array([5, 10, 15, 20, 25])

# Fancy indexing:
# NumPy looks at indexes array and picks values at those positions
print(arr[indexes])  # [10 20]

# arr[indexes]
=> # indexes = [1, 3]
=> # NumPy:
     => # goes to index 1 → value 10
     => # goes to index 3 → value 20


🔹 Step 4: Selecting Rows & Columns Together (2D Fancy Indexing)

import numpy as np 
# Create a 2D NumPy array (3 rows × 3 columns)
arr = np.array([
    [1, 2, 3],   # Row 0
    [4, 5, 6],   # Row 1
    [7, 8, 9]    # Row 2
])

# Fancy indexing with rows & columns together:
# [0, 2]  → row positions
# [1, 2]  → column positions
# Pairing happens like:
# (0,1) and (2,2)
result = arr[[0, 2], [1, 2]]

# Print selected values
print(result) # [2 9]

# What is happening (easy words)
=> # Step 1: Understand the array with indexes

        Col 0  Col 1  Col 2
Row 0 →   1      2      3
Row 1 →   4      5      6
Row 2 →   7      8      9

=> # Step 2: This line is the key

arr[[0, 2], [1, 2]]

# Let’s break it into two parts.

=> # First list → ROW selection
[0, 2]

# Note: This means:
=> # Pick row 0
=> # Pick row 2

Row 0 → [1, 2, 3]
Row 2 → [7, 8, 9]

=> # Second list → COLUMN selection
[1, 2]

# Note: This means:
=> # From the selected rows, look at:
    => # column 1
    => # column 2

# VERY IMPORTANT RULE → Pairwise selection
# NumPy now pairs both lists by position, not by combination.

Row list:    [0,  2]
Column list: [1,  2]
               ↓   ↓
Pairs:      (0,1) (2,2)

# How NumPy picks values (step-by-step)

# First pair
arr[0, 1]  # Row 0, Column 1 → 2

# Second pair
arr[2, 2]  # Row 2, Column 2 → 9


=> # First list [0, 2] → rows to pick
=> # Second list [1, 2] → columns to pick
=> # NumPy pairs them position by position


Step 3: Pairing happens like this

| Row index | Column index | Value |
| --------- | ------------ | ----- |
| 0         | 1            | 2     |
| 2         | 2            | 9     |


🔹 Step 5: Boolean Indexing (Most Important)
=> # What is Boolean Indexing?
=> # Select elements using True/False conditions

# Basic Boolean Condition

arr = np.array([10, 20, 30, 40, 50])

# Create a boolean mask
mask = arr > 25

print(mask)   # [False False  True  True  True]

🔹 Step 6: Conditional Selection (Multiple Conditions)
# AND Condition

arr = np.array([10, 20, 30, 40, 50])

# Select values greater than 20 AND less than 50
result = arr[(arr > 20) & (arr < 50)]

print(result)

# OR Condition

# Select values less than 20 OR greater than 40
result = arr[(arr < 20) | (arr > 40)]

print(result)

# Always use:

=> # & instead of and
=> # | instead of or

🔹 Step 7: Boolean Indexing in 2D Arrays

import numpy as np 
arr = np.array([
    [5, 10, 15],
    [20, 25, 30],
    [35, 40, 45]
])

# Select all elements greater than 20
print(arr[arr > 20])  # [25 30 35 40 45]

🔹 Step 8: Filtering Rows Using Conditions
Example: Select rows where first column > 10

import numpy as np  # Import NumPy

# Create a 2D NumPy array
arr = np.array([
    [5, 100],    # Row 0
    [15, 200],   # Row 1
    [25, 300]    # Row 2
])

# Step 1: Select all rows where first column (column 0) > 10
# arr[:, 0] → selects all rows in column 0 → [5, 15, 25]
# arr[:, 0] > 10 → checks condition for each value → [False, True, True]
# Using this boolean array to filter rows: arr[boolean_array]
rows = arr[arr[:, 0] > 10]

# Step 2: Print filtered rows
print(rows)

# Output 
[[ 15 200]
 [ 25 300]]

# Step-by-step explanation (simple)
=> # 1. arr[:, 0] → pick all rows of column 0
[5, 15, 25]

=> # 2. arr[:, 0] > 10 → compare each value with 10
[False, True, True]

# Visual understanding

Original array:
[[  5 100]   ← first column 5 → skipped
 [ 15 200]   ← first column 15 → included
 [ 25 300]]  ← first column 25 → included

# Output:
[[ 15 200]
 [ 25 300]]

🔹 Step 9: Replacing Values Conditionally

import numpy as np
arr = np.array([10, 20, 30, 40, 50])
arr[arr > 30] = 999
print(arr)


Replace negative values with 0

import numpy as np
arr = np.array([10, -5, 20, -3, 30])
arr[arr < 0] = 0
print(arr)

🔹 Step 10: np.where() – Conditional Replace & Select
=> # Replace using np.where

import numpy as np 

# Create a NumPy array
arr = np.array([10, 20, 30, 40])

# np.where(condition, value_if_true, value_if_false)
# Check each value in arr:
# - If value > 25 → put 1
# - Else → put 0
result = np.where(arr > 25, 1, 0)

# Print the result array
print(result)   # [0 0 1 1]

🔹 Step 11: np.nonzero() – Get Index Positions

import numpy as np  # Import NumPy

# Create a NumPy array
arr = np.array([0, 10, 0, 20, 30])

# np.nonzero():
# It finds positions (indexes) where value is NOT zero
indexes = np.nonzero(arr)

# Print the indexes
print(indexes)  # (array([1, 3, 4]),)

🔹 Step 12: Difference Between Slicing & Fancy Indexing

import numpy as np  # Import NumPy

# Original NumPy array
arr = np.array([1, 2, 3, 4, 5])

# SLICING:
# arr[1:4] creates a VIEW (it shares memory with original array)
slice_arr = arr[1:4]

# FANCY INDEXING:
# arr[[1, 2]] creates a COPY (new memory)
fancy_arr = arr[[1, 2]]

# Change all values inside the sliced array
slice_arr[:] = 99

# Print original array
# It changes because slice_arr is a VIEW
print(arr)  # [ 1 99 99 99  5 ]

# Print fancy indexed array
# It does NOT change because it is a COPY
print(fancy_arr)  # [2 3]

# What is happening (very simple)
🔹 Slicing (arr[1:4])
=> # Does not create a new array
=> # It points to the same memory

arr        → [1  2  3  4  5]
slice_arr →     [2  3  4]   (same memory)


🔹 Fancy indexing (arr[[1, 2]])

=> # Creates a new array
=> # Uses separate memory
=> # Changing original does not affect it

fancy_arr → [2  3]  (new memory)

Key difference (must remember)

| Feature              | Slicing | Fancy Indexing |
| -------------------- | ------- | -------------- |
| Uses range (`:`)     | ✅       | ❌              |
| Uses list of indexes | ❌       | ✅              |
| Returns view         | ✅       | ❌              |
| Returns copy         | ❌       | ✅              |
| Affects original     | ✅       | ❌              |


# Note: One-line memory trick

=> # Colon (:) = view → changes original
=> # List ([]) = copy → safe
 
```
📌 7. Shape Manipulation in NumPy
```python
✅ Requirements
# Before starting, you should know:
=> # What shape means
=> # 1D, 2D arrays
=> # Basic indexing

🔹 1. reshape() – Change Shape Without Changing Data
Purpose:
=> # Change the shape of an array
=> # Total number of elements must stay the same

import numpy as np 

# Create a 1D NumPy array with 6 elements
arr = np.array([1, 2, 3, 4, 5, 6])

# Reshape the 1D array into:
# 2 rows and 3 columns
# NOTE: Total elements must stay the same (2 × 3 = 6)
new_arr = arr.reshape(2, 3)

# Print the reshaped array
print(new_arr)

# output:
[[1 2 3]
 [4 5 6]]

🔹 Example 2: Using -1 (Auto calculate dimension)

import numpy as np  

# Create a 1D NumPy array
arr = np.array([1, 2, 3, 4, 5, 6])

# Reshape with -1:
# 3 rows given
# -1 tells NumPy: "calculate columns automatically"
new_arr = arr.reshape(3, -1)

# Print the reshaped array
print(new_arr)


# output:
[[1 2]
 [3 4]
 [5 6]]

# What -1 means (very easy)
=> # -1 = "You decide for me, NumPy"
=> # NumPy looks at:
     => # Total elements = 6
     => # Rows = 3
=> # So columns = 6 ÷ 3 = 2

# Important rules to remember

1. Total elements must match
reshape(2, 4)  # ❌ Error (8 needed, only 6 exist)

2. Only one -1 is allowed
reshape(-1, -1)  # ❌ Not allowed

3. reshape usually returns a VIEW
  => # Changing reshaped array may change original

# One-line memory trick
reshape = same data, new shape

🔹 3. flatten() – Convert to 1D (Copy)
Purpose:
=> # Convert any array into 1D
=> # Always returns a copy


import numpy as np  
arr = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

flat = arr.flatten()
print(flat)

🔹 3. ravel() – Convert to 1D (View if Possible)
Purpose:

=> # Similar to flatten()
=> # Returns a view when possible (memory efficient)

import numpy as np

# Create a 2D array
arr = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

# Convert 2D array into 1D (flatten-like)
rav = arr.ravel()

# Modify the first element of ravel array
rav[0] = 99

# Original array also changes (because ravel gives a VIEW)
print(arr)

# output
[[99  2  3]
 [ 4  5  6]]

# Note: ravel() → view → changes original (if possible)

🔹 4. transpose() – Swap Rows & Columns
Purpose:
=> # Converts rows into columns
=> # Used in matrix math

import numpy as np

# Create a 2D array
arr = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

# Swap rows and columns
trans = arr.transpose()
print(trans)

# Output:
[[1 4]
 [2 5]
 [3 6]]

# Shape change

print(arr.shape)    # (2, 3) → 2 rows, 3 columns
print(trans.shape)  # (3, 2) → 3 rows, 2 columns

# Note: transpose = rows ↔ columns

🔹 5. .T – Shortcut for Transpose

# Same as arr.transpose()
print(arr.T)

🔹 6. swapaxes() – Swap Any Two Axes
Purpose
=> # Swap specific axes
=> # Mostly used in 3D or higher arrays

Swap specific axes (useful in 3D+ arrays)

import numpy as np
arr = np.array([
    [[1, 2], [3, 4]],
    [[5, 6], [7, 8]]
])

# Swap axis 0 and axis 1
swapped = arr.swapaxes(0, 1)
print(swapped)

🔹 expand_dims() – Add a new dimension

import numpy as np 

# Create a 1D array (only one row of values)
arr = np.array([1, 2, 3])

# Add axis at position 0 (make it a ROW)

# Add a new dimension at axis 0
# axis=0 means: add a new outer level
# This turns the 1D array into a single row
new_arr = np.expand_dims(arr, axis=0)

# Print the new array
print(new_arr)

# Print the shape (rows, columns)
print(new_arr.shape)


# Output
[[1 2 3]]
(1, 3)

# Add axis at position 1 (make it a COLUMN)

# Add a new dimension at axis 1
# axis=1 means: add a new inner level
# This turns each value into its own row (column shape)
new_arr = np.expand_dims(arr, axis=1)

# Print the new array
print(new_arr)

# Print the shape
print(new_arr.shape)


# Output
[[1]
 [2]
 [3]]
(3, 1)

# Visual comparison

Original
[1  2  3]

axis=0 (row)
[[1  2  3]]

axis=1 (column)
[[1]
 [2]
 [3]]

# Why this is useful (real reason)

=> # Machine Learning models expect 2D or 3D data
=> # expand_dims() helps match required shape

No data is changed, only shape

# Easy memory trick

axis=0 → add row
axis=1 → add column

🔹 8. squeeze() – Remove Single-Dimension Axes
Purpose:
=> # Remove axes of size 1

arr = np.array([[[1, 2, 3]]])

print(arr.shape)

squeezed = arr.squeeze()

print(squeezed)
print(squeezed.shape)


🔹 9. Understanding Shape Compatibility (IMPORTANT)

Rule:
Total number of elements must remain same during reshape

arr = np.array([1, 2, 3, 4, 5, 6])

#  This will cause an error
# arr.reshape(4, 2)

#  Correct
arr.reshape(2, 3)

🔹 10. Common Shape Transform Examples

# Convert 1D to Row Vector
arr = np.array([1, 2, 3])

row = arr.reshape(1, -1)

print(row)

# Convert 1D to Column Vector

col = arr.reshape(-1, 1)

print(col)


🔹 11. View vs Copy Summary

| Method        | Returns            |
| ------------- | ------------------ |
| `reshape()`   | View (if possible) |
| `ravel()`     | View (if possible) |
| `flatten()`   | Copy               |
| `transpose()` | View               |
| `swapaxes()`  | View               |


# Quick Cheat Sheet

| Task             | Method          |
| ---------------- | --------------- |
| Change shape     | `reshape()`     |
| Make 1D copy     | `flatten()`     |
| Make 1D view     | `ravel()`       |
| Transpose        | `.T`            |
| Swap axes        | `swapaxes()`    |
| Add dimension    | `expand_dims()` |
| Remove dimension | `squeeze()`     |


```

