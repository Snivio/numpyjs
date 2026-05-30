# @snivio/numpy

NumPy, but for TypeScript. No Python runtime, no WASM blobs, no nonsense. Just typed arrays and math.

If you've ever wished you could `np.linalg.solve()` in a Node script without spinning up a Jupyter notebook, this is for you.

---

## Install

```bash
npm install @snivio/numpy
```

Works in Node 16+, Bun, Deno, and any bundler (Vite, webpack, esbuild). Ships ESM + CJS + full `.d.ts` types.

---

## Quick Start

```typescript
import np, { array, zeros, arange, linspace } from "@snivio/numpy";

// vectors & matrices
const v = array([1, 2, 3, 4, 5]);
const m = arange(0, 12).reshape([3, 4]);

// element-wise math (broadcasting just works)
const doubled = np.multiply(v, 2);
const normed = np.divide(v, np.sqrt(np.sum(np.power(v, 2))));

// linear algebra
const A = array([
  [2, 1],
  [5, 7],
]);
const b = array([11, 13]);
const x = np.linalg.solve(A, b); // solves Ax = b

// statistics
np.mean(v); // 3
np.std(v); // 1.414...
np.corrcoef(v, v); // [[1, 1], [1, 1]]

// random (seeded for reproducibility)
np.random.seed(42);
const noise = np.random.randn(100, 100); // 100x100 gaussian noise
```

---

## What's Inside

| Module           | Highlights                                                                   |
| ---------------- | ---------------------------------------------------------------------------- |
| **Core**         | N-dim typed arrays, strides, broadcasting, slicing with steps                |
| **Creation**     | `zeros`, `ones`, `full`, `arange`, `linspace`, `eye`, `diag`, `tile`         |
| **Math**         | All the trig, exp/log, floor/ceil, clip, element-wise comparisons            |
| **Linalg**       | `det`, `inv`, `solve`, `eig`, `svd`, `qr`, `cholesky`, `norm`                |
| **Stats**        | `mean`, `median`, `std`, `var`, `corrcoef`, `cov`, `histogram`               |
| **Random**       | Xorshift128+ PRNG, `normal`, `uniform`, `poisson`, `gamma`, `beta`, `choice` |
| **FFT**          | Cooley-Tukey radix-2: `fft`, `ifft`, `fft2`, `fftfreq`, `fftshift`           |
| **Manipulation** | `concat`, `stack`, `split`, `flip`, `rot90`, `unique`, `where`, `meshgrid`   |

---

## API Cheat Sheet

### Array Creation

```typescript
np.array([
  [1, 2],
  [3, 4],
]); // from nested arrays
np.zeros([3, 3]); // 3×3 of zeros
np.ones([2, 4], "float32"); // with explicit dtype
np.arange(0, 10, 0.5); // [0, 0.5, 1, ..., 9.5]
np.linspace(0, 1, 50); // 50 points from 0 to 1
np.eye(4); // 4×4 identity
np.diag([1, 2, 3]); // diagonal matrix
```

### Math (element-wise)

```typescript
np.add(a, b)        np.subtract(a, b)
np.multiply(a, b)   np.divide(a, b)
np.power(a, 2)      np.sqrt(a)
np.sin(a)           np.exp(a)
np.log(a)           np.clip(a, 0, 1)
np.abs(a)           np.round(a)
```

### Linear Algebra

```typescript
np.dot(a, b); // dot product / matmul
np.linalg.det(A); // determinant
np.linalg.inv(A); // inverse
np.linalg.solve(A, b); // solve linear system
np.linalg.eig(A); // eigenvalues & vectors
np.linalg.svd(A); // singular value decomposition
np.linalg.norm(v); // L2 norm
np.linalg.qr(A); // QR decomposition
np.linalg.cholesky(A); // Cholesky factorization
```

### Statistics

```typescript
np.mean(a)          np.median(a)
np.std(a)           np.var_(a)
np.min(a)           np.max(a)
np.sum(a, axis)     np.prod(a, axis)
np.corrcoef(x, y)   np.cov(x, y)
np.percentile(a, 75)
np.histogram(a, 10)
```

### Random

```typescript
np.random.seed(123); // reproducible
np.random.rand(3, 3); // uniform [0, 1)
np.random.randn(3, 3); // standard normal
np.random.randint(0, 100, [5]); // random ints
np.random.normal(0, 1, [1000]); // gaussian
np.random.choice(a, 5); // sample
np.random.shuffle(a); // in-place shuffle
```

### Manipulation

```typescript
a.reshape([2, 6])         a.transpose()
a.flatten()               a.slice([...])
np.concatenate([a, b])    np.stack([a, b])
np.flip(a)                np.rot90(a)
np.unique(a)              np.where(cond, x, y)
np.diff(a)                np.cumsum(a)
np.meshgrid(x, y)         np.outer(a, b)
```

---

## Dtypes

Backed by real TypedArrays, no boxing overhead.

`float64` (default) · `float32` · `int32` · `int16` · `int8` · `uint32` · `uint16` · `uint8` · `bool`

```typescript
const a = np.zeros([1000, 1000], "float32"); // ~4MB instead of ~8MB
const b = a.astype("int32");
```

---

## Why This Exists

Python's NumPy is fantastic but sometimes you just need to crunch numbers in a TypeScript codebase without crossing language boundaries. This library gives you the same mental model (shapes, broadcasting, axis-based reductions) in a package you can `import` and go.

Not a 1:1 port (no C extensions here), but covers the 90% of NumPy that most people actually use day-to-day.

---

## Contributing

PRs welcome. Run the test suite before submitting:

```bash
npm test        # 68 tests via vitest
npm run lint    # eslint
npm run build   # tsup, outputs cjs + esm + .d.ts
```

---

## License

MIT
