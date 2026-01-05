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

🔹 1️⃣ Select a FULL column

# Select ALL rows, but only column at index 1 (second column)
print(arr_2d[:, 1])

# What is happening (simple meaning)
=> : → take all rows
=> 1 → take column index 1
=> Result → one full column

# Output
=> # [20 50 80]

🔹 2️⃣ Select MULTIPLE columns

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

```

