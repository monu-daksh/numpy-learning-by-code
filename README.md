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
📌 8. Copy vs View in NumPy
```python
✅ Requirements
Before starting, you should know:
=> # Basic indexing & slicing
=> # What is a NumPy array
=> # That arrays live in memory

🔹 1. What is a View?
=> # A view is:
=> # A new array object
=> Shares the same memory with the original array

import numpy as np
# Create a NumPy array
arr = np.array([10, 20, 30, 40, 50])

# Take a slice from index 1 to 3
# This does NOT create a new array
# It creates a VIEW (shared memory)
view_arr = arr[1:4]

print(view_arr)   # [20 30 40]


=> # Modify the view
# Change the first element of the view
view_arr[0] = 999

# Print original array
print(arr)

# Output: [10 999 30 40 50]

What is really happening? (Very simple)
=> # arr[1:4] does NOT copy data
=> # NumPy does not create a new array
=> # It just points to the same memory
=> # This is called a VIEW

# Memory trick
=> # Slice = Same memory
=> # Fancy index = New memory

🔹 2. What is a Copy
Simple Definition:
A copy is:

=> # A completely new array
=> # Has its own memory

=> # Changing copy does NOT affect original

Example: Copy Creation

import numpy as np

# Create a NumPy array
arr = np.array([10, 20, 30, 40, 50])

# Create a COPY of the array
# .copy() makes a new array with its OWN memory
copy_arr = arr.copy()

# Change the first element of the copied array
copy_arr[0] = 111

# Print original array
# Original array is NOT affected
print(arr)

# Print copied array
# Only this array is changed
print(copy_arr)

=> # What is happening (very simple)
# .copy() creates a new array
# Data is duplicated
# Memory is different
# No connection to original array

# Changing copy_arr does NOT affect arr(original arr)
Because:
=> # arr and copy_arr
=> # do NOT share memory

Think like this
=> # .copy() = photocopy of data

Easy memory rule
=> # slice → shared memory (view)
=> # slice → shared memory (view)

🔹 3. How Slicing Creates Views (IMPORTANT)

import numpy as np

# Create a NumPy array
arr = np.array([1, 2, 3, 4, 5])

# Take a slice from index 1 to 3
# This creates a VIEW (not a new array)
slice_arr = arr[1:4]

# Change ALL values inside the slice
slice_arr[:] = 0

# Print the original array
print(arr)  # [1 0 0 0 5]

# What is happening
=> # arr[1:4] creates a VIEW
=> # No new data is created
=> # NumPy only points to part of arr
=> # Both arrays share the same memory

# slice_arr[:] = 0
=> # : means all elements of the slice
=> # You replace every selected value with 0

🔹 4. When NumPy Creates Copies Automatically
# Fancy Indexing → Copy

arr = np.array([10, 20, 30, 40])

fancy = arr[[0, 2]]

fancy[0] = 999

print(arr)    # Original unchanged
print(fancy)

# Boolean Indexing → Copy

arr = np.array([10, 20, 30, 40])
filtered = arr[arr > 20]
filtered[:] = 0
print(arr)       # Original unchanged
print(filtered)

🔹 5. copy() Method (Best Practice)
# When to use?
=> # When you do NOT want side effects
=> # When modifying sliced data safely

arr = np.array([5, 10, 15, 20])

safe_copy = arr[1:3].copy()

safe_copy[:] = 99

print(arr)
print(safe_copy)


🔹 6. Memory Behavior (base attribute)

# Check if array is a view or copy

arr = np.array([1, 2, 3, 4])

view_arr = arr[1:3]
copy_arr = arr[1:3].copy()

print(view_arr.base)  # Not None → View
print(copy_arr.base)  # None → Copy


🔹 7. View vs Copy Summary Table

| Operation        | Result             |
| ---------------- | ------------------ |
| Slicing (`:`)    | View               |
| `reshape()`      | View (if possible) |
| `ravel()`        | View (if possible) |
| Fancy indexing   | Copy               |
| Boolean indexing | Copy               |
| `copy()`         | Copy               |
| `flatten()`      | Copy               |


🔹 8. Common Mistakes (VERY IMPORTANT)
# Mistake 1: Modifying Slice Thinking It’s Safe

import numpy as np

# Original NumPy array
arr = np.array([1, 2, 3, 4])

# Take a slice (this creates a VIEW, not a copy)
temp = arr[1:3]

# Change all values in the slice
temp[:] = 100

# Print original array
# Original array changes too (unexpected for beginners)
print(arr)  # [  1 100 100   4]

# What went wrong?
# 1. arr[1:3] creates a VIEW
=> # temp does NOT have its own data
=> # It shares memory with arr

Indexes selected:
arr = [1, 2, 3, 4]
          ↑  ↑
        temp slice

# 2. temp[:] = 100
=> # : means change everything inside temp
=> # Since memory is shared
      => # arr[1] becomes 100
      => # arr[2] becomes 100

# Correct Way: Use .copy()

# Take a slice AND make a copy
temp = arr[1:3].copy()

# Modify the copy
temp[:] = 100

# Original array stays unchanged
print(arr) # [1 2 3 4]

# Easy rules to remember
=> # Slice → View → shared memory
=> # View change → original changes
=> # Use .copy() when editing safely

# Mistake 2: Boolean Filter Modify Original

filtered = arr[arr > 2]
filtered[:] = 0

print(arr)  # No change (people expect change)

Explanation:
=> # Boolean indexing creates a copy

🔹 9. Visual Memory Explanation (Simple Words)

Original Array Memory
┌────────────────────┐
│ 10  20  30  40  50 │
└────────────────────┘
      ↑   ↑   ↑
      View (slice)

# Copy creates:

New Memory
┌────────────────────┐
│ 20  30  40         │
└────────────────────┘


🔹 10. Best Practices (IMPORTANT)
✔ Use slicing for performance
✔ Use .copy() for safety
✔ Always assume slice = view
✔ Use base to debug memory behavior
```
📌 9. Joining & Splitting Arrays
```python
✅ Requirements
=> # Before starting, you should know:
      => # 1D and 2D arrays
      => # What rows and columns mean
      => # shape of an array

🔹 1. np.concatenate() (Base Function – Most Important)

# What it does:
=> # Joins arrays along an existing axis
=> # This is the core function (others are wrappers)

# ----------------Example 1: Join 1D arrays --------------------------

import numpy as np

# Create first 1D array
a = np.array([1, 2, 3])

# Create second 1D array
b = np.array([4, 5, 6])

# Join (concatenate) both arrays
# Arrays are joined one after another (end-to-end)
result = np.concatenate((a, b))

# Print the joined array
print(result)  # [1 2 3 4 5 6]

Example 2: Join 2D arrays

import numpy as np

# --------- Example 1: Join 2D arrays ROW-WISE ---------

# First 2D array (2 rows, 2 columns)
a = np.array([
    [1, 2],
    [3, 4]
])

# Second 2D array (1 row, 2 columns)
b = np.array([
    [5, 6]
])

# axis = 0 → join by ROWS (top to bottom)
# Number of COLUMNS must be same in both arrays
row_result = np.concatenate((a, b), axis=0)

print("Row-wise join:")
print(row_result)


# --------- Example 2: Join 2D arrays COLUMN-WISE ---------

# First 2D array (2 rows, 2 columns)
a = np.array([
    [1, 2],
    [3, 4]
])

# Second 2D array (2 rows, 1 column)
b = np.array([
    [5],
    [6]
])

# axis = 1 → join by COLUMNS (left to right)
# Number of ROWS must be same in both arrays
col_result = np.concatenate((a, b), axis=1)
print(col_result)

🔹 axis = 0 (ROW-WISE)
=> # Adds new rows
=> # Stacks arrays vertically
=> # Columns must match

[1 2]        [1 2]
[3 4]  +     [3 4]
             [5 6]


🔹 axis = 1 (COLUMN-WISE)
=> # Adds new columns
=> # Stacks arrays horizontally
=> # Rows must match

# Visual:
[1 2]    [5]    [1 2 5]
[3 4] +  [6] →  [3 4 6]


🔹 2. np.vstack() (Vertical Stack)
=> # vstack means “put one array on top of another”

=> v = vertical
=> vertical = top to bottom


# Visual:

=> # Imagine you have two rows of numbers written on paper:

Row 1: 1  2  3
Row 2: 4  5  6

=> # If you stack them vertically, you place one row below the other:

1  2  3
4  5  6

That is exactly what np.vstack() does.


=> # Example 1: 1D Arrays (most common confusion)

import numpy as np

# First 1D array (just one row)
a = np.array([1, 2, 3])

# Second 1D array (another row)
b = np.array([4, 5, 6])

# Stack a and b vertically (top to bottom)
result = np.vstack((a, b))

print(result)  
[[1 2 3]
 [4 5 6]]

=> # Example 2: 2D Arrays

import numpy as np

# First 2D array (1 row, 2 columns)
a = np.array([[1, 2]])

# Second 2D array (1 row, 2 columns)
b = np.array([[3, 4]])

# Stack vertically
result = np.vstack((a, b))

print(result)


=> # Shape understanding (important)

print(a.shape)        # (3,)
print(result.shape)  # (2, 3)

=> # Easy memory trick

| Function | Meaning                               |
| -------- | ------------------------------------- |
| `vstack` | Stack **vertically** (top → bottom)   |
| `hstack` | Stack **horizontally** (left → right) |


🔹 3. np.hstack() (Horizontal Stack)
=> # np.hstack() puts arrays side-by-side (left → right)

=> # horizontal
=> # horizontal = left to right

Think like this (real-life)
Imagine numbers written like this:

Array A: 1  2  3
Array B: 4  5  6

If you join them side-by-side, you get:

1  2  3  4  5  6

Example 1: 1D Arrays

import numpy as np

# First 1D array
a = np.array([1, 2, 3])

# Second 1D array
b = np.array([4, 5, 6])

# Join a and b horizontally (side-by-side)
result = np.hstack((a, b))

# Print the final result
print(result)  # [1 2 3 4 5 6]


=> # Shape understanding (important)

print(a.shape)       # (3,)
print(result.shape) # (6,)

=> # Example 2: 2D Arrays

import numpy as np

# First 2D array (2 rows, 1 column)
a = np.array([[1],
              [2]])

# Second 2D array (2 rows, 1 column)
b = np.array([[3],
              [4]])

# Join a and b horizontally (column-wise)
result = np.hstack((a, b))

# Print the result
print(result)


[[1 3]
 [2 4]]

=> # What is happening visually

Before stacking:

a =      b =
1        3
2        4

After hstack:

1  3
2  4

=> Columns are added
=> Number of rows stays the same

=> # Important rule (VERY IMPORTANT)
For np.hstack():

✅ Number of rows must be same
❌ This will fail:

a = np.array([[1], [2], [3]])
b = np.array([[4], [5]])

np.hstack((a, b))

Because:
=> # a has 3 rows
=> # b has 2 rows

=> # Easy memory trick

| Function | Meaning                                    |
| -------- | ------------------------------------------ |
| `vstack` | Stack **top to bottom** (rows increase)    |
| `hstack` | Stack **left to right** (columns increase) |

One-line summary (memorize this)
np.hstack() joins arrays side-by-side, creating more columns.

🔹 4. np.dstack() (Depth Stack – 3D)

=> # np.dstack() stacks arrays one behind another (depth-wise)

=> #  d = depth
=> # depth = front ↔ back (3rd dimension)

"""
Imagine transparent sheets of paper:
Sheet A (numbers written)
Sheet B (numbers written)

You place one sheet on top of another
Not left-right, not up-down — front-back
"""

# First understand the shape
Your arrays:

a = [[1, 2],
     [3, 4]]

b = [[5, 6],
     [7, 8]]

Each one is 2 rows × 2 columns

a =        b =
1  2       5  6
3  4       7  8

=> # Code with VERY SIMPLE comments

import numpy as np

# First 2D array (2 rows, 2 columns)
a = np.array([[1, 2],
              [3, 4]])

# Second 2D array (same shape as a)
b = np.array([[5, 6],
              [7, 8]])

# Stack a and b along the depth (3rd axis)
result = np.dstack((a, b))

# Print the result
print(result)

# Print the shape (rows, columns, depth)
print(result.shape)

# Output

[[[1 5]
  [2 6]]

 [[3 7]
  [4 8]]]

(2, 2, 2)

=> # What is ACTUALLY happening?
"""
Before dstack → 2D arrays

Each array is flat like a page.

After dstack → 3D array

Now every cell has depth.

Let’s look at one position:

"""

result[0][0]

👉 That position contains:

[1, 5]

Meaning:

1 came from array a

5 came from array b


Visual explanation
=> # Think of result like this:

=> # Depth layer 0 (a)

1  2
3  4


=> # Depth layer 1 (b)

5  6
7  8

=> # So NumPy stores it as:

[
  [ [1,5], [2,6] ],
  [ [3,7], [4,8] ]
]

# Shape explained

print(result.shape)

(2, 2, 2)

Means:

2 rows

2 columns

2 depth layers

🔹 5. np.column_stack() (Column Wise – Smart)
It takes 1D arrays and turns them into columns of a table

Real-life example
Imagine you have this data:

a → Roll numbers

b → Marks

Roll   Marks
1      4
2      5
3      6

=> # Example

import numpy as np

# First 1D array (will become column 1)
a = np.array([1, 2, 3])

# Second 1D array (will become column 2)
b = np.array([4, 5, 6])

# Convert both arrays into columns and join them
result = np.column_stack((a, b))

# Print the final 2D array
print(result)


What is happening step by step?

Step 1: Original arrays (1D)
a = [1, 2, 3]
b = [4, 5, 6]

Step 2: Convert into columns
a → 1    b → 4
     2        5
     3        6

Step 3: Join side by side
[[1 4]
 [2 5]
 [3 6]]

# Shape explained simply
(3, 2)

Means:
=> # 3 rows
=> # 2 columns

Why not hstack?
Let’s compare.

=> # hstack

np.hstack((a, b)) # [1 2 3 4 5 6]
Loses table structure

column_stack
np.column_stack((a, b))

[[1 4]
 [2 5]
 [3 6]]

Perfect table / dataset format

Why column_stack() is VERY useful

Used when:
=> Creating datasets
=> ML features
=> CSV-like data
=> Pandas DataFrame input

X = np.column_stack((age, salary, experience))
Each array = one feature column


Important rule

=> # All arrays must have:
   => # Same number of elements
=> # Otherwise ❌ error.

🔹 6. np.row_stack() (Row Wise)

What it does:
=> # Converts 1D arrays into rows
=> # Same as vstack()


a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

result = np.row_stack((a, b))

print(result)

🔹 7. Key Differences (VERY IMPORTANT)
| Function       | Direction | Axis         |
| -------------- | --------- | ------------ |
| `concatenate`  | Any       | User-defined |
| `vstack`       | Rows      | axis=0       |
| `hstack`       | Columns   | axis=1       |
| `dstack`       | Depth     | axis=2       |
| `column_stack` | Columns   | Smart        |
| `row_stack`    | Rows      | Smart        |


🔹 8. Common Mistakes (Learn This)
=> # Shape mismatch

a = np.array([[1, 2]])
b = np.array([[3, 4, 5]])

# np.vstack((a, b))  ❌ Error

Column counts must match for row stacking


Quick Cheat Sheet

np.concatenate((a, b), axis=0)  # rows
np.concatenate((a, b), axis=1)  # columns
np.vstack((a, b))               # row stack
np.hstack((a, b))               # column stack
np.dstack((a, b))               # depth stack
np.column_stack((a, b))         # columns
np.row_stack((a, b))            # rows



🔹 9. np.split() (Base Function – Most Important)
np.split() cuts one array into smaller pieces

Think:
=> # Like cutting a chocolate bar 
=> # Or slicing a list into parts

Two ways np.split() works

=> # np.split() can split an array in 2 different ways:
=> # Split into equal parts
=> # Split at specific index positions

Example 1: Split into equal parts

import numpy as np

# Original 1D array
arr = np.array([10, 20, 30, 40, 50, 60])

# Split the array into 3 equal parts
result = np.split(arr, 3)

# Print the result
print(result)  # [array([10, 20]), array([30, 40]), array([50, 60])]

# Important rule
=> # Array length must be divisible by number of parts

Common Error (why it fails)
np.split(arr, 4)

Error because:
=> Array length = 6
=> 6 ÷ 4  (not possible)

Example 2: Split using index positions (MOST USED)

import numpy as np

# Original array
arr = np.array([1, 2, 3, 4, 5, 6, 7])

# Split at index 2 and index 5
result = np.split(arr, [2, 5])

print(result)

What does [2, 5] mean?

It means:

Cut BEFORE index 2
Cut BEFORE index 5


Index map

Index:  0  1  | 2  3  4 | 5  6
Value:  1  2  | 3  4  5 | 6  7

# Output

[
 array([1, 2]),       # arr[0:2]
 array([3, 4, 5]),    # arr[2:5]
 array([6, 7])        # arr[5:end]
]

Very important thing to understand
Index number itself is NOT included in the left part
(Same rule as Python slicing)
arr[0:2]  # includes index 0,1


Example 3: Single index split

arr = np.array([10, 20, 30, 40, 50])

result = np.split(arr, [3])

print(result) # [array([10, 20, 30]), array([40, 50])]  # [10, 20, 30 | 40, 50]


Example 4: Split into every element (edge case)

arr = np.array([1, 2, 3])

result = np.split(arr, 3)

print(result)  # [array([1]), array([2]), array([3])]

Real-life analogy (very helpful)

Roll numbers = [1, 2, 3, 4, 5, 6]

You want:

Batch 1 → first 2 students

Batch 2 → next 2

Batch 3 → last 2


np.split(arr, 3)
Perfect use case.

🔹 2. np.hsplit() (Horizontal Split – Columns)
It splits an array into parts column-wise (left → right)

Example (2D array)

import numpy as np

# 2D array (2 rows, 4 columns)
arr = np.array([[1, 2, 3, 4],
                [5, 6, 7, 8]])

# Split array into 2 parts horizontally (by columns)
result = np.hsplit(arr, 2)

# Print the result
print(result)


[
 array([[1, 2],
        [5, 6]]),

 array([[3, 4],
        [7, 8]])
]


# What happened
=> # Original columns = 4
=> # Split into 2 parts
=> # Each part gets 2 columns

[1 2 | 3 4]
[5 6 | 7 8]

Rule (important)
=> # Number of columns must be divisible by splits (This fails)
np.hsplit(arr, 3)


🔹 3. np.vsplit() (Vertical Split – Rows)
It splits an array by rows

Example 1: Equal row split

import numpy as np

# 2D array (4 rows, 2 columns)
arr = np.array([
    [1, 2],
    [3, 4],
    [5, 6],
    [7, 8]
])

# Split array into 2 equal parts by rows
result = np.vsplit(arr, 2)

# Print result
print(result)

# Output

[
 array([[1, 2],
        [3, 4]]),

 array([[5, 6],
        [7, 8]])
]

=> # What happened
=> # Total rows = 4
=> # Split into 2 parts
=> # Each part gets 2 rows

[1 2]
[3 4]
------
[5 6]
[7 8]


Rule (important)
=> # Rows must be divisible by number of splits(This gives error)
np.vsplit(arr, 3)

Example 2: Split using row index

# Split after row index 1 and 3
result = np.vsplit(arr, [1, 3])

print(result)

# What this means
Split points:
=> # After row 1
=> # After row 3

So parts become:
arr[0:1]   → first row
arr[1:3]   → middle rows
arr[3:end] → last rows

# output:
[
 array([[1, 2]]),

 array([[3, 4],
        [5, 6]]),

 array([[7, 8]])
]

# Easy memory trick
| Function   | Cuts                   |
| ---------- | ---------------------- |
| `hsplit()` | Columns (left → right) |
| `vsplit()` | Rows (top → bottom)    |

🔹 4. np.dsplit() (Depth Split – 3D Arrays)

What it does:
=> # Splits arrays along third axis
=> # Used in image processing (RGB channels)

Example: Split a 3D array

arr = np.array([[[1, 2, 3],
                 [4, 5, 6]],
                
                [[7, 8, 9],
                 [10, 11, 12]]])

print(arr.shape)


result = np.dsplit(arr, 3)
print(result)


# Explanation:
=> # Original shape: (2, 2, 3)
=> # After split → 3 arrays, each with shape (2, 2, 1)
=> # Each split represents one depth layer

🔹 5. Axis Understanding (Very Important)
| Function   | Splits Along | Axis         |
| ---------- | ------------ | ------------ |
| `split()`  | Any axis     | User defined |
| `hsplit()` | Columns      | axis=1       |
| `vsplit()` | Rows         | axis=0       |
| `dsplit()` | Depth        | axis=2       |


🔹 6. Real-Life Example (ML / Data Science)
# Splitting features and labels

data = np.array([[1, 85],
                 [2, 90],
                 [3, 95]])

# Split into features and labels
features, labels = np.hsplit(data, 2)

print(features)
print(labels)


🔹 7. Common Mistakes (Very Important)
# Unequal splits

arr = np.array([1, 2, 3, 4, 5])
np.split(arr, 2)  # ❌ Error

# Correct way
np.split(arr, [2])

# Quick Cheat Sheet
np.split(arr, 3)           # Equal split
np.split(arr, [2, 5])      # Index split
np.hsplit(arr, 2)          # Column split
np.vsplit(arr, 2)          # Row split
np.dsplit(arr, 3)          # Depth split

# One-Line Memory Trick
=> # h → horizontal → columns
=> # v → vertical → rows
=> # d → depth → 3D
```
📌 10. Mathematical Operations
```python

🔹 1. Element-Wise Operations

import numpy as np

a = np.array([10, 20, 30])
b = np.array([1, 2, 3])

result = a + b
print(result)  # [11 22 33]

# Explanation:

=> # 10 + 1
=> # 20 + 2
=> # 30 + 3
Each index is calculated individually

🔹 2. Addition (+ / np.add())

a = np.array([5, 10, 15])
b = np.array([2, 3, 4])

# Using + operator
print(a + b)

# Using NumPy function
print(np.add(a, b))
# Both give the same result

🔹 3. Subtraction (- / np.subtract())

a = np.array([10, 20, 30])
b = np.array([3, 5, 7])

print(a - b)
print(np.subtract(a, b))


# Explanation:
=> # 10 − 3
=> # 20 − 5
=> # 30 − 7

🔹 4. Multiplication (* / np.multiply())

a = np.array([2, 4, 6])
b = np.array([3, 5, 7])

print(a * b)
print(np.multiply(a, b))   # [ 6 20 42 ]

# It multiplies:
2 * 3 = 6
4 * 5 = 20
6 * 7 = 42

So the result is:  [6, 20, 42]


🔹 5. Division (/ / np.divide())

import numpy as np

a = np.array([10, 20, 30])
b = np.array([2, 5, 3])

print(a / b)    # [5. 4. 10.]
print(np.divide(a, b))  # [5. 4. 10.]

# Note: Always returns float values

🔹 6. Floor Division (// / np.floor_divide())

a = np.array([10, 20, 25])
b = np.array([3, 6, 4])

print(a // b)    # [3 3 6]
print(np.floor_divide(a, b)) # [3 3 6]

=> # NumPy divides element-by-element, then:
=> # removes the decimal part by rounding DOWN

10 / 3 = 3.333 → floor → 3
20 / 6 = 3.333 → floor → 3
25 / 4 = 6.25  → floor → 6

# So result becomes:
[3, 3, 6]


🔹 7. Modulus (% / np.mod())
import numpy as np

a = np.array([10, 20, 25])
b = np.array([3, 6, 4])

print(a % b)   # [1 2 1]
print(np.mod(a, b)) # [1 2 1]

# What is happening?
=> # NumPy does element-wise modulus.
=> # It gives remainder after division:

10 % 3 = 1   (10 = 3×3 + 1)
20 % 6 = 2   (20 = 6×3 + 2)
25 % 4 = 1   (25 = 4×6 + 1)

🔹 8. Power (** / np.power())

import numpy as np

a = np.array([2, 3, 4])

# Square each element
print(a ** 2)  # [ 4  9 16]

# Cube each element
print(np.power(a, 3))  # [ 8 27 64]

🔹 9. Unary Operations (Single Array)

import numpy as np

a = np.array([10, -20, 30])

print(-a)  # [-10  20 -30]

# What is happening?
NumPy changes the sign of each element:

10   → -10
-20  →  20
30   → -30

=> # Positive becomes negative
=> # Negative becomes positive

# Important thing
=> # This does NOT change the original array
=> # It creates a new array

print(a)   # still same


# Absolute Value (abs() / np.abs())
a = np.array([-10, -20, 30])

print(abs(a))
print(np.abs(a))

# Explanation:
=> # Converts negative numbers to positive

🔹 10. Rounding (round() / np.round())

import numpy as np

a = np.array([1.2, 2.6, 3.5, 4.9])

print(np.round(a))  # [1. 3. 4. 5.]

# Explanation:
=> # Rounds to nearest whole number

🔹 11. Floor (np.floor())

import numpy as np

a = np.array([1.9, 2.1, 3.7])

print(np.floor(a))  # [1. 2. 3.]

# Explanation:
=> # Always rounds down

🔹 12. Ceil (np.ceil())
# np.ceil() is the opposite of np.floor()

import numpy as np

a = np.array([1.1, 2.3, 3.2])

print(np.ceil(a))  # [2. 3. 4.]

🔹 13. Operations with Scalars
a = np.array([10, 20, 30])

print(a + 5)    # Add 5 to every element
print(a * 2)    # Multiply each element by 2

# Note: This is called broadcasting

🔹 14. Real-Life Example (Marks Increase)

import numpy as np

# Original marks of students
marks = np.array([70, 80, 90])

# Add 5 grace marks to every student
updated_marks = marks + 5

print(updated_marks) # [75 85 95]

# Quick Cheat Sheet

a + b          # Addition
a - b          # Subtraction
a * b          # Multiplication
a / b          # Division
a // b         # Floor division
a % b          # Modulus
a ** 2         # Power
np.abs(a)      # Absolute
np.round(a)    # Round
np.floor(a)    # Floor
np.ceil(a)     # Ceil
```

📌 11. Universal Functions (ufuncs)
```python
# What are ufuncs? (In very easy words)

=> # ufuncs = functions that work on every element automatically


# * Instead of doing this:

# Normal Python (slow + messy)
result = []
for x in [1, 4, 9]:
    result.append(x ** 0.5)

# * NumPy lets you do this:

import numpy as np

arr = np.array([1, 4, 9])
print(np.sqrt(arr))

# What happened?
# => sqrt() ran on each element
# => No loop
=> # Very fast
 This is a ufunc


 🔹 Why ufuncs are special
=> # Work element by element
=> # Much faster than Python loops
=> # Code looks clean and short

🔹 Trigonometric Functions (sin, cos, tan)
=> # NumPy uses radians, not degrees.
# Angle conversions:

| Degrees | Radians |
| ------- | ------- |
| 0°      | 0       |
| 90°     | π / 2   |
| 180°    | π       |
| 360°    | 2π      |

# That’s why we use np.pi/2 instead of 90.

Example: sin(), cos()

import numpy as np
# import → bring something from outside
# numpy → a powerful math library
# np → short name (alias) for numpy

angles = np.array([0, np.pi/2, np.pi])
# angles → a variable name
# np.array → creates a NumPy array (like a list, but smarter)
# [0, np.pi/2, np.pi] → values inside the array
#
# 0         → 0 radians (0°)
# np.pi/2  → π divided by 2 → 90°
# np.pi    → π → 180°

print(np.sin(angles))
# np.sin → sine function from NumPy
# angles → NumPy applies sin() to EACH value
# print → shows the result on screen

print(np.cos(angles))
# np.cos → cosine function from NumPy
# angles → cosine of EACH value

# What is np.pi?
np.pi = 3.141592653589793

# => What does NumPy do internally?
What does NumPy do internally?
# => np.sin(angles) means:
np.sin(0)        # sin(0°) --> = 0
np.sin(np.pi/2) # sin(90°) --> = 1
np.sin(np.pi)   # sin(180°) --> 0


# => np.cos(angles) means:
np.cos(0)        # cos(0°) --> = 1
np.cos(np.pi/2) # cos(90°) --> = 0
np.cos(np.pi)   # cos(180°) --> -1

# Note:-
# => NumPy automatically loops over the array
# => NumPy automatically loops over the array

# print(np.sin(angles))  -->  [0. 1. 0.]
# print(np.cos(angles))  --> [ 1.  0. -1.]

# => Why dots after numbers? (0. 1.)
Because NumPy uses floating-point numbers (decimals).
# => 1. = 1.0
# => 0. = 0.0

#=> Very Simple Real-Life Example
Imagine:

# => Angles = directions
# => sin() = vertical movement
# => cos() = horizontal movement

At 90°:
# => You move straight up
# => Horizontal = 0
# => Vertical = 1

🔹Convert degrees to radians (easy way)
degrees = np.array([0, 30, 60, 90])

radians = np.deg2rad(degrees)

print(np.sin(radians))

🔹 Exponential Functions
=> # First: What is e?
=> # e is a special mathematical number

Value ≈ 2.71828

=> # Just like:
   => # π (pi) ≈ 3.14
   => # e ≈ 2.718

Exmaple:
import numpy as np
# import → bring numpy library
# numpy → used for math and arrays
# np → short name for numpy

arr = np.array([1, 2, 3])
# arr → a NumPy array
# contains numbers: 1, 2, 3

print(np.exp(arr))
# np.exp → exponential function
# exp(x) means → e raised to the power x
# NumPy applies exp() to EACH value in arr

=> # What EXACTLY happens inside?
This line:
np.exp(arr)

Is equivalent to:
[e¹, e², e³]

Which means:
[2.718¹, 2.718², 2.718³]

Output (approximate)
[ 2.71828183  7.3890561  20.08553692 ]


# Why does NumPy do this automatically?
Because NumPy is vectorized:

# => You give it many values at once
# => It applies the function to each value
# => No loops needed

# Example 2 — Single Number
print(np.exp(1))  # e¹ = 2.718

# Example 3 — Including 0 and Negative Numbers
arr = np.array([-1, 0, 1])
print(np.exp(arr))

Calculation:

| Input | Calculation | Result  |
| ----- | ----------- | ------- |
| -1    | e⁻¹         | ≈ 0.367 |
| 0     | e⁰          | 1       |
| 1     | e¹          | ≈ 2.718 |

# e⁰ = 1 (important rule)

# Example 4 — Manual vs NumPy
# Without NumPy (slow way):

import math

print(math.exp(1))
print(math.exp(2))
print(math.exp(3))

# With NumPy (fast way):
np.exp([1, 2, 3])

# NumPy = cleaner + faster

# Why is exp() used in Machine Learning?
1️⃣ Probability (Softmax)

scores = np.array([1, 2, 3])
exp_scores = np.exp(scores)
probabilities = exp_scores / np.sum(exp_scores)

print(probabilities)

# Converts raw scores into probabilities

2️⃣ Logistic Regression
Formula: 1 / (1 + e⁻ˣ)

3️⃣ Growth & Decay
# => Population growth
# => Interest calculation
# => Neural network activations

Very Simple Real-Life Meaning
Think of exp() as FAST growth

| x  | exp(x)   |
| -- | -------- |
| 1  | small    |
| 2  | bigger   |
| 3  | very big |
| 10 | HUGE     |

🔹 Logarithmic Functions
# => Logarithmic Functions

A logarithm answers this question: "How much power do you need?"

#Example:
# => log₂(8) = 3 → because 2³ = 8
# => log₁₀(100) = 2 → because 10² = 100

#So:
# => log = reverse of power

# NumPy Log Functions (What they mean)

| Function     | Meaning                  |
| ------------ | ------------------------ |
| `np.log()`   | Natural log (base **e**) |
| `np.log10()` | Log base **10**          |
| `np.log2()`  | Log base **2**           |

# What is e?
# => e is a special math number ≈ 2.718
# => Used in ML, growth, decay, science
# => Natural log = log base e

🔹 Comparison Functions (True / False output)
# Example: greater than, equal to

arr = np.array([10, 20, 30])

print(arr > 15)
print(arr == 20)

# Output:
# => True if condition matches
# => False otherwise

# Using NumPy functions
# => print(np.greater(arr, 15))
# => print(np.equal(arr, 20))

# Maximum & Minimum
a = np.array([10, 20, 30])
b = np.array([15, 18, 25])

print(np.maximum(a, b))
print(np.minimum(a, b))

🔹 Bitwise Functions (Binary level)
# Used when working with bits.

AND
a = np.array([1, 2, 3])
b = np.array([3, 2, 1])

print(np.bitwise_and(a, b))

OR
print(np.bitwise_or(a, b))

XOR
print(np.bitwise_xor(a, b))

NOT
print(np.bitwise_not(a))

🔹 More Common ufuncs (Must Know)
Square root

arr = np.array([4, 9, 16])
print(np.sqrt(arr))

Square
print(np.square(arr))

Absolute value
arr = np.array([-10, 20, -30])
print(np.abs(arr))

Limit values (clip)
arr = np.array([5, 15, 25])
print(np.clip(arr, 10, 20))

🔹 Performance Benefit (Why ufuncs are fast )
Python way (slow)

result = []
for x in arr:
    result.append(x * 2)

NumPy ufunc way (fast)
print(arr * 2)

```
📌 12. Broadcasting (Core Concept)
```python
What is Broadcasting? (Simple Meaning)
# Broadcasting = NumPy automatically adjusts shapes so operations can work
# “NumPy repeats smaller data to match bigger data — without copying it”

Example

import numpy as np
arr = np.array([1, 2, 3])
print(arr + 10)

What happened?
# => 10 is applied to every element
# => NumPy broadcasted 10 → [10, 10, 10]

🔹 Why Broadcasting is Useful
# Without broadcasting:

# You would need a loop (slow)
result = []
for x in arr:
    result.append(x + 10)

# With broadcasting:
print(arr + 10)

🔹 Broadcasting Rules (VERY IMPORTANT)
# NumPy checks shapes from right to left.

# Two dimensions are compatible if:
# => 1. They are equal, OR
# => 2. One of them is 1

🔹 1. Scalar Broadcasting (Most Common)
# Scalar + Array

arr = np.array([5, 10, 15])
result = arr * 2
print(result)

🔹 2. Vector Broadcasting (1D + 2D)
#Adding a row vector to a matrix

matrix = np.array([[1, 2, 3],
                   [4, 5, 6]])

vector = np.array([10, 20, 30])
result = matrix + vector
print(result)

# A matrix is a table of numbers (rows x columns)
# Here: 2 rows, 3 columns
matrix = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

# A vector is a single list of numbers
# Length = 3 (same as number of columns in matrix)
vector = np.array([10, 20, 30])

🔹 3. Column Broadcasting (Using reshape)
import numpy as np

# A matrix (2 rows, 3 columns)
matrix = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

# A simple list of numbers
# reshape(2, 1), (2 rows, 1 cloumn) turns it into a column (vertical)
column = np.array([10, 20]).reshape(2, 1)

# NumPy automatically adds:
# 10 to the first row
# 20 to the second row
result = matrix + column

print(result)

🔹 4. Matrix Broadcasting (2D + 2D)
# Same shape → works

import numpy as np

a = np.array([[1, 2],
              [3, 4]])

b = np.array([[10, 20],
              [30, 40]])

print(a + b)


# One dimension is 1 → still works
a = np.array([[1, 2, 3],
              [4, 5, 6]])

b = np.array([[10, 20, 30]])

print(a + b)

🔹 5. Broadcasting with Different Shapes (Rule in Action)
a = np.array([[1],
              [2],
              [3]])

b = np.array([10, 20, 30])

print(a + b)

# Explanation:
# => a shape → (3,1)
# => b shape → (3,)
# => NumPy expands them to (3,3)

🔹 6. Common Broadcasting Errors
Shape mismatch

a = np.array([1, 2, 3])
b = np.array([10, 20])

# print(a + b)  Error

# Why?
# => Shapes (3,) and (2,) cannot match

🔹 7. How to Fix Broadcasting Errors
import numpy as np

# Example matrix (2 rows, 2 columns)
matrix = np.array([
    [1, 2],
    [3, 4]
])

#  This will cause a broadcasting error
# Because shape (2,) does NOT clearly match rows or columns
# b = np.array([10, 20])
# result = matrix + b


#  FIX 1: Using reshape()
# reshape(1, 2) makes it a ROW vector
# Shape becomes (1 row, 2 columns)
b_row = np.array([10, 20]).reshape(1, 2)

# NumPy adds [10, 20] to EACH row of the matrix
result_row = matrix + b_row
print("Using reshape (row):")
print(result_row)


#  FIX 2: Using newaxis
# [:, np.newaxis] turns array into a COLUMN vector
# Shape becomes (2 rows, 1 column)
b_col = np.array([10, 20])[:, np.newaxis]

# NumPy adds:
# 10 to first row
# 20 to second row
result_col = matrix + b_col
print(result_col)


🔹 8. Broadcasting with Functions (ufuncs)

arr = np.array([1, 4, 9])

print(np.sqrt(arr + 10))

# Explanation:
# => 10 is broadcasted
# => Then sqrt() applied to each element

🔹 9. Real-World Use Cases

Example 1: Increase salaries

salary = np.array([30000, 40000, 50000])
updated_salary = salary + 5000
print(updated_salary)

Example 2: Normalize marks
marks = np.array([70, 80, 90])
percentage = (marks / 100) * 100
print(percentage)

Example 3: Add bias in ML
weights = np.array([[0.5, 0.2, 0.1],
                    [0.4, 0.6, 0.3]])

bias = np.array([0.1, 0.2, 0.3])

output = weights + bias

print(output)


🔹 10. Broadcasting Visual Summary
(3, 3) + (3,) → OK
(3, 3) + (1, 3) → OK
(3, 3) + (3, 1) → OK
(3,)   + (2,)   → ❌

Golden Rule (Remember This)
# If shapes match from right to left → broadcasting works

# Quick Cheat Sheet
arr + 5               # scalar broadcasting
matrix + vector       # row broadcasting
matrix + column       # column broadcasting
reshape()             # fix shape issues
np.newaxis            # add dimension
```
📌 13. Aggregation & Statistical Functions
```python
What are Aggregation Functions?
# => Take many values
# => Return one summary value
# => Example: total, average, minimum, maximum

🔹 Sample Array (We’ll use this everywhere)

import numpy as np
arr = np.array([10, 20, 30, 40, 50])

🔹 1. sum() → Total of all values

total = np.sum(arr)
print(total) // 150

🔹 2. min() & max() → Smallest & largest value

# min() finds the smallest number
print(np.min(arr))   # Output: 10

# max() finds the biggest number
print(np.max(arr))   # Output: 50

🔹 3. mean() → Average value
# Formula: (sum of values) / (number of values)
# (10 + 20 + 30 + 40 + 50) / 5 = 150 / 5 = 30

print(np.mean(arr)) // 30.0

🔹 4. median() → Middle value
# Sorted array: [10, 20, 30, 40, 50]
# Middle value is 30

print(np.median(arr)) // 30.0

🔹 5. std() → Standard Deviation
# Measures how much numbers differ from the average
# Bigger value = numbers are more spread out

print(np.std(arr)) // 14.142135623730951

🔹 6. var() → Variance
# Variance = (standard deviation)²
# Shows spread of data in squared form

print(np.var(arr)) // 200.0

🔹 7. argmin() & argmax() → Index of min/max
arr2 = np.array([10, 5, 30, 2, 50])

# Smallest number is 2 → its position (index) is 3
print(np.argmin(arr2))
# Output: 3

# Largest number is 50 → its position (index) is 4
print(np.argmax(arr2))
# Output: 4

🔹 8. Axis-Based Calculations (VERY IMPORTANT)
 matrix = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

# Sum of ALL numbers
# 1+2+3+4+5+6 = 21
print(np.sum(matrix))
# Output: 21


# axis=0 → work column-wise
# Column sums:
# [1+4, 2+5, 3+6]
print(np.sum(matrix, axis=0))
# Output: [5 7 9]


# axis=1 → work row-wise
# Row sums:
# [1+2+3, 4+5+6]
print(np.sum(matrix, axis=1))
# Output: [ 6 15 ]


# Column-wise minimum
# Compare values in each column
print(np.min(matrix, axis=0))
# Output: [1 2 3]


# Row-wise maximum
print(np.max(matrix, axis=1))
# Output: [3 6]

🔹 9. Mean with Axis
 matrix = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

# Column averages:
# [(1+4)/2, (2+5)/2, (3+6)/2]
print(np.mean(matrix, axis=0))
# Output: [2.5 3.5 4.5]

# Row averages:
# [(1+2+3)/3, (4+5+6)/3]
print(np.mean(matrix, axis=1))
# Output: [2. 5.]

🔹 10. Cumulative Operations (Running Total)
arr3 = np.array([10, 20, 30, 40])

# cumsum() → running total
# [10,
#  10+20,
#  10+20+30,
#  10+20+30+40]
print(np.cumsum(arr3))
# Output: [ 10  30  60 100 ]


# cumprod() → running multiplication
# [10,
#  10*20,
#  10*20*30,
#  10*20*30*40]
print(np.cumprod(arr3))
# Output: [    10    200   6000 240000 ]


# Column-wise cumulative sum
print(np.cumsum(matrix, axis=0))
# Output:
# [[1 2 3]
#  [5 7 9]]

# Row-wise cumulative sum
print(np.cumsum(matrix, axis=1))
# Output:
# [[ 1  3  6]
#  [ 4  9 15]]

```
📌 14. Sorting & Searching
```python

import numpy as np
arr = np.array([30, 10, 50, 20, 40])

🔹 1. sort() → Sort values
Basic sorting (ascending)

# np.sort() returns a NEW sorted array (original stays same)
sorted_arr = np.sort(arr)

print(sorted_arr)
# Output: [10 20 30 40 50]

# Original array is NOT changed
print(arr)
# Output: [30 10 50 20 40]


# Sort IN-PLACE (original array changes)
arr.sort()
print(arr)
# Output: [10 20 30 40 50]

🔹 2. argsort() → Sort indices
arr = np.array([30, 10, 50, 20])

# argsort returns indices that would sort the array
indices = np.argsort(arr)

print(indices)
# Output: [1 3 0 2]

# Explanation:
# index 1 → value 10 (smallest)
# index 3 → value 20
# index 0 → value 30
# index 2 → value 50

# Use indices to get sorted array
print(arr[indices])
# Output: [10 20 30 50]

🔹 3. Sorting Along Axis (2D Arrays)
matrix = np.array([
    [3, 1, 2],
    [6, 4, 5]
])

# Sort each ROW (axis=1 → across rows)
print(np.sort(matrix, axis=1))
# Output:
# [[1 2 3]
#  [4 5 6]]

# Sort each COLUMN (axis=0 → down columns)
print(np.sort(matrix, axis=0))
# Output:
# [[3 1 2]
#  [6 4 5]]

matrix = np.array([
    [3, 1, 2],
    [6, 4, 5]
])

# Sort each ROW (axis=1 → across rows)
print(np.sort(matrix, axis=1))
# Output:
# [[1 2 3]
#  [4 5 6]]

# Sort each COLUMN (axis=0 → down columns)
print(np.sort(matrix, axis=0))
# Output:
# [[3 1 2]
#  [6 4 5]]

🔹 4. where() → Find values using condition
arr = np.array([10, 20, 30, 40, 50])

# Find indices where condition is TRUE
result = np.where(arr > 30)

print(result)
# Output: (array([3, 4]),)

# Get actual values using indices
print(arr[result])
# Output: [40 50]


# Replace values using where()
# If value > 30 → put 100
# Else → keep original value
new_arr = np.where(arr > 30, 100, arr)

print(new_arr)
# Output: [10 20 30 100 100]

🔹 5. searchsorted() → Find insert position
arr = np.array([10, 20, 30, 40])  # MUST be sorted

# Find where 25 should be inserted
index = np.searchsorted(arr, 25)

print(index)
# Output: 2
# Means: 25 fits between 20 and 30

# Insert multiple values
values = [15, 35]
print(np.searchsorted(arr, values))
# Output: [1 3]

🔹 6. Conditional Searching (Boolean Indexing)
arr = np.array([10, 15, 20, 25, 30])

# Select values greater than 20
filtered = arr[arr > 20]

print(filtered)
# Output: [25 30]

# Combine conditions
# Select values > 10 AND < 30
filtered = arr[(arr > 10) & (arr < 30)]

print(filtered)
# Output: [15 20 25]

🔹 7. unique() → Get unique values
arr = np.array([1, 2, 2, 3, 3, 4, 5])

# Remove duplicates
unique_values = np.unique(arr)

print(unique_values)
# Output: [1 2 3 4 5]

# Get unique values with their counts
values, counts = np.unique(arr, return_counts=True)

print(values)
# Output: [1 2 3 4 5]

print(counts)
# Output: [1 2 2 1 1]

```
📌 15. Random Module
```python
# What is the Random Module?
# => Used to:
# => Generate random numbers
# => Simulate real-world randomness
# => Create test data

🔹 1. Random Numbers Basics
import numpy as np
print(np.random.rand())

Note: Generate a random float between 0 and 1

Meaning:
Gives a random decimal number
Range → 0 (inclusive) to 1 (exclusive)

🔹 2. rand() → Random floats (uniform distribution)
# 1D array with 5 random values
arr = np.random.rand(5)
print(arr)


# Output: [0.95 0.73 0.60 0.15 0.15]
# All values are between 0 and 1

Explanation:
Creates 5 random numbers
All values are between 0 and 1

# 2D array → 2 rows, 3 columns
matrix = np.random.rand(2, 3)

print(matrix)
# Output:
# [[0.15 0.05 0.86]
#  [0.60 0.70 0.02]]


🔹 3. randn() → Random numbers from normal distribution

# Random numbers around mean = 0
# Can be positive or negative
arr = np.random.randn(5)

print(arr)
# Output: [-0.23  1.55 -0.98  0.45 -0.12]


# 2D normal distribution
matrix = np.random.randn(2, 3)

print(matrix)
# Output:
# [[-0.46  0.38  1.12]
#  [ 0.89 -1.23  0.15]]

🔹 4. randint() → Random integers
# Random integers from 1 to 9 (10 excluded)
arr = np.random.randint(1, 10, size=5)

print(arr)
# Output: [3 7 1 9 4]


# 2D random integers
matrix = np.random.randint(0, 100, size=(2, 3))

print(matrix)
# Output:
# [[45 12 89]
#  [67 34 21]]

🔹 5. choice() → Random selection
arr = np.array([10, 20, 30, 40])

# Pick ONE random value
print(np.random.choice(arr))
# Output: 30


# Pick MULTIPLE random values
print(np.random.choice(arr, size=3))
# Output: [40 10 30]


# Pick with PROBABILITY
# Higher probability = higher chance
print(np.random.choice(arr, size=3, p=[0.1, 0.2, 0.3, 0.4]))
# Output example: [40 30 40]

🔹 6. shuffle() → Shuffle array in-place
arr = np.array([1, 2, 3, 4, 5])

# Shuffles original array
np.random.shuffle(arr)

print(arr)
# Output: [3 5 1 2 4]
# Original order is changed

🔹 7. permutation() → Shuffled copy
arr = np.array([1, 2, 3, 4, 5])

# Returns a NEW shuffled array
new_arr = np.random.permutation(arr)

print(new_arr)
# Output: [2 5 1 4 3]

print(arr)
# Output: [1 2 3 4 5]
# Original array stays same

🔹 8. Setting Random Seed (VERY IMPORTANT)
# Seed makes random numbers repeatable
np.random.seed(42)

print(np.random.rand(3))
# Output: [0.37454012 0.95071431 0.73199394]

# Run again → SAME output every time 

🔹 9. Probability Distributions
# NORMAL DISTRIBUTION
# loc = mean, scale = standard deviation
arr = np.random.normal(loc=0, scale=1, size=5)

print(arr)
# Output: [ 0.24 -0.91  1.46 -0.22  0.07]


# UNIFORM DISTRIBUTION
# Values between 10 and 20
arr = np.random.uniform(low=10, high=20, size=5)

print(arr)
# Output: [13.2 18.6 11.4 19.8 15.3]


# BINOMIAL DISTRIBUTION
# n = number of trials
# p = probability of success
arr = np.random.binomial(n=10, p=0.5, size=5)

print(arr)
# Output: [4 6 5 7 3]
# Simulates coin toss results
```
📌 15. Linear Algebra in NumPy
```python

🔹 1. Dot Product
# Multiply elements and add them
# [a1, a2] · [b1, b2] = a1*b1 + a2*b2

import numpy as np

a = np.array([1, 2])
b = np.array([3, 4])

# Dot product of two vectors
result = np.dot(a, b)

print(result)

🔹 2. Matrix Multiplication (NOT element-wise)
# This is different from *

# Uses rows of A and columns of B

A = np.array([[1, 2],
              [3, 4]])

B = np.array([[5, 6],
              [7, 8]])

# Matrix multiplication using @
result = A @ B

print(result)
# Output:
# [[19 22]
#  [43 50]]
# Explanation:
# Row1 × Col1 → (1*5 + 2*7) = 19
# Row1 × Col2 → (1*6 + 2*8) = 22

🔹 3. matmul() (Same as @)
result = np.matmul(A, B)

print(result)
# Output is SAME as A @ B

🔹 4. Determinant
# What is Determinant?
# => A single number that tells:
# => If matrix is invertible
# => Area scaling factor

# A single number describing the matrix
# If determinant = 0 → inverse not possible

A = np.array([[1, 2],
              [3, 4]])

det = np.linalg.det(A)

print(det)
# Output: -2.0

Note: If determinant = 0 → no inverse

🔹 5. Inverse Matrix
# Inverse cancels the original matrix
# A × A⁻¹ = Identity matrix

A_inv = np.linalg.inv(A)

print(A_inv)
# Output:
# [[-2.   1. ]
#  [ 1.5 -0.5]]

# Verify correctness
print(A @ A_inv)
# Output (almost identity):
# [[1. 0.]
#  [0. 1.]]

🔹 6. Identity Matrix
# Like number 1 for matrices

I = np.eye(3)

print(I)
# Output:
# [[1. 0. 0.]
#  [0. 1. 0.]
#  [0. 0. 1.]]

🔹 7. Eigenvalues & Eigenvectors
# Special values where direction stays same after transformation

# Eigenvector direction does NOT change
# Eigenvalue tells how much it stretches

A = np.array([[4, 2],
              [1, 3]])

values, vectors = np.linalg.eig(A)

print("Eigenvalues:")
print(values)

print("Eigenvectors:")
print(vectors)

# Check eigen rule: A @ v = λ * v
v = vectors[:, 0]       # first eigenvector
lambda_val = values[0] # first eigenvalue

print(A @ v)
print(lambda_val * v)
# Both outputs should match

🔹 8. Solving Linear Equations

# Example:
# 2x + y = 5
# x + 3y = 6

A = np.array([[2, 1],
              [1, 3]])

B = np.array([5, 6])

solution = np.linalg.solve(A, B)

print(solution)
# Output: [1.8 1.4]
# solution[0] → x
# solution[1] → y


🔹 9. Element-wise vs Matrix multiplication (Important)

print(A * B)
# Element-wise multiplication
# Each value multiplies directly

print(A @ B)
# Matrix multiplication
# Rows × Columns

```




