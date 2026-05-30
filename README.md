# numpyjs

A comprehensive NumPy-like numerical computing library for TypeScript/JavaScript.

## Installation

```bash
npm install numpyjs
```

## Features

- **N-dimensional arrays (NdArray)** - Core data structure with full shape/stride support
- **Array creation** - zeros, ones, full, empty, arange, linspace, eye, identity, etc.
- **Mathematical operations** - Element-wise arithmetic, trigonometry, exponential, logarithmic
- **Broadcasting** - NumPy-compatible broadcasting for operations on arrays of different shapes
- **Indexing & slicing** - Advanced indexing with support for ranges and steps
- **Shape manipulation** - reshape, transpose, flatten, squeeze, expand_dims, concatenate, stack
- **Linear algebra** - dot, matmul, inv, det, eig, svd, solve, norm, qr
- **Statistics** - mean, median, std, var, min, max, sum, prod, percentile, corrcoef
- **Random** - uniform, normal, randint, choice, shuffle, various distributions
- **FFT** - fft, ifft, fft2, ifft2, fftfreq
- **Sorting & searching** - sort, argsort, argmin, argmax, where, nonzero
- **Dtype system** - float32, float64, int8, int16, int32, uint8, uint16, uint32, bool

## Usage

```typescript
import np from "numpyjs";

// Create arrays
const a = np.array([1, 2, 3, 4, 5]);
const b = np.zeros([3, 3]);
const c = np.arange(0, 10, 0.5);
const d = np.linspace(0, 1, 100);

// Math operations
const sum = np.add(a, a);
const product = np.multiply(a, np.array([2, 2, 2, 2, 2]));

// Linear algebra
const matrix = np.array([
  [1, 2],
  [3, 4],
]);
const det = np.linalg.det(matrix);
const inv = np.linalg.inv(matrix);

// Statistics
const mean = np.mean(a);
const std = np.std(a);

// Random
const random = np.random.randn(3, 3);

// Reshape
const reshaped = np.arange(0, 12).reshape([3, 4]);
```

## API Reference

### Array Creation

- `np.array(data, dtype?)` - Create array from nested arrays
- `np.zeros(shape, dtype?)` - Array of zeros
- `np.ones(shape, dtype?)` - Array of ones
- `np.full(shape, value, dtype?)` - Array filled with value
- `np.empty(shape, dtype?)` - Uninitialized array
- `np.arange(start?, stop, step?, dtype?)` - Evenly spaced values
- `np.linspace(start, stop, num?, dtype?)` - Linearly spaced values
- `np.eye(n, m?, k?, dtype?)` - Identity matrix
- `np.identity(n, dtype?)` - Identity matrix
- `np.diag(v, k?)` - Diagonal matrix or extract diagonal

### Math Operations

- `np.add`, `np.subtract`, `np.multiply`, `np.divide`
- `np.power`, `np.sqrt`, `np.abs`, `np.exp`, `np.log`
- `np.sin`, `np.cos`, `np.tan`, `np.arcsin`, `np.arccos`, `np.arctan`
- `np.floor`, `np.ceil`, `np.round`, `np.clip`

### Linear Algebra (`np.linalg`)

- `np.linalg.dot`, `np.linalg.matmul`
- `np.linalg.inv`, `np.linalg.det`
- `np.linalg.eig`, `np.linalg.svd`
- `np.linalg.solve`, `np.linalg.norm`
- `np.linalg.qr`, `np.linalg.cholesky`

### Statistics

- `np.mean`, `np.median`, `np.std`, `np.var`
- `np.min`, `np.max`, `np.sum`, `np.prod`
- `np.percentile`, `np.corrcoef`, `np.cov`

### Random (`np.random`)

- `np.random.rand`, `np.random.randn`
- `np.random.randint`, `np.random.uniform`
- `np.random.normal`, `np.random.choice`
- `np.random.shuffle`, `np.random.seed`

## License

MIT
