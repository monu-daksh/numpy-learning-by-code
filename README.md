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

