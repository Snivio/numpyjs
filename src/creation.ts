/**
 * Array creation routines — zeros, ones, arange, linspace, eye, etc.
 */

import { NdArray, NestedArray } from "./core/ndarray";
import { DtypeString, createTypedArray, getDtypeInfo } from "./core/dtype";
import {
  Shape,
  shapeSize,
  validateShape,
  flatToMulti,
  multiToFlat,
  computeStrides,
} from "./core/shape";

// --- Helpers ---

const flattenNested = (data: NestedArray): { flat: number[]; shape: Shape } => {
  if (typeof data === "number" || typeof data === "boolean")
    return { flat: [Number(data)], shape: [] };

  const shape: Shape = [];
  let cur: NestedArray = data;
  while (Array.isArray(cur)) {
    shape.push(cur.length);
    cur = cur[0];
  }

  const flat: number[] = [];
  const walk = (arr: NestedArray): void => {
    if (typeof arr === "number" || typeof arr === "boolean") {
      flat.push(Number(arr));
      return;
    }
    for (const item of arr) walk(item);
  };
  walk(data);
  return { flat, shape };
};

// --- Public API ---

export const array = (
  data: NestedArray | number[],
  dtype: DtypeString = "float64",
): NdArray => {
  const { flat, shape } = flattenNested(data);
  const buf = createTypedArray(dtype, flat.length);
  for (let i = 0; i < flat.length; i++) buf[i] = flat[i];
  return new NdArray(buf, shape, dtype);
};

export const zeros = (
  shape: Shape | number,
  dtype: DtypeString = "float64",
): NdArray => {
  const s = typeof shape === "number" ? [shape] : shape;
  validateShape(s);
  return new NdArray(createTypedArray(dtype, shapeSize(s)), s, dtype);
};

export const ones = (
  shape: Shape | number,
  dtype: DtypeString = "float64",
): NdArray => {
  const s = typeof shape === "number" ? [shape] : shape;
  validateShape(s);
  const data = createTypedArray(dtype, shapeSize(s));
  data.fill(1);
  return new NdArray(data, s, dtype);
};

export const full = (
  shape: Shape | number,
  value: number,
  dtype: DtypeString = "float64",
): NdArray => {
  const s = typeof shape === "number" ? [shape] : shape;
  validateShape(s);
  const data = createTypedArray(dtype, shapeSize(s));
  data.fill(value);
  return new NdArray(data, s, dtype);
};

export const empty = (
  shape: Shape | number,
  dtype: DtypeString = "float64",
): NdArray => {
  const s = typeof shape === "number" ? [shape] : shape;
  validateShape(s);
  return new NdArray(createTypedArray(dtype, shapeSize(s)), s, dtype);
};

export const arange = (
  start: number,
  stop?: number,
  step?: number,
  dtype: DtypeString = "float64",
): NdArray => {
  if (stop === undefined) {
    stop = start;
    start = 0;
  }
  if (step === undefined) step = 1;
  if (step === 0) throw new Error("step must not be zero");

  const len = Math.max(0, Math.ceil((stop - start) / step));
  const data = createTypedArray(dtype, len);
  for (let i = 0; i < len; i++) data[i] = start + i * step;
  return new NdArray(data, [len], dtype);
};

export const linspace = (
  start: number,
  stop: number,
  num = 50,
  endpoint = true,
  dtype: DtypeString = "float64",
): NdArray => {
  if (num < 0) throw new Error("num must be non-negative");
  if (num === 0) return new NdArray(createTypedArray(dtype, 0), [0], dtype);
  if (num === 1) {
    const d = createTypedArray(dtype, 1);
    d[0] = start;
    return new NdArray(d, [1], dtype);
  }

  const step = endpoint ? (stop - start) / (num - 1) : (stop - start) / num;
  const data = createTypedArray(dtype, num);
  for (let i = 0; i < num; i++) data[i] = start + i * step;
  return new NdArray(data, [num], dtype);
};

export const logspace = (
  start: number,
  stop: number,
  num = 50,
  endpoint = true,
  base = 10,
  dtype: DtypeString = "float64",
): NdArray => {
  const lin = linspace(start, stop, num, endpoint, dtype);
  const data = createTypedArray(dtype, num);
  for (let i = 0; i < num; i++) data[i] = base ** lin.data[i];
  return new NdArray(data, [num], dtype);
};

export const eye = (
  N: number,
  M?: number,
  k = 0,
  dtype: DtypeString = "float64",
): NdArray => {
  const cols = M ?? N;
  const data = createTypedArray(dtype, N * cols);
  for (let i = 0; i < N; i++) {
    const j = i + k;
    if (j >= 0 && j < cols) data[i * cols + j] = 1;
  }
  return new NdArray(data, [N, cols], dtype);
};

export const identity = (n: number, dtype: DtypeString = "float64"): NdArray =>
  eye(n, n, 0, dtype);

export const diag = (v: NdArray | number[], k = 0): NdArray => {
  const arr = v instanceof NdArray ? v : array(v);

  if (arr.ndim === 1) {
    const n = arr.size + Math.abs(k);
    const out = zeros([n, n], arr.dtype);
    for (let i = 0; i < arr.size; i++) {
      const row = k >= 0 ? i : i - k;
      const col = k >= 0 ? i + k : i;
      if (row < n && col < n) out.data[row * n + col] = arr.data[i];
    }
    return out;
  }

  if (arr.ndim === 2) {
    const [rows, cols] = arr.shape;
    const len = Math.min(
      k >= 0 ? Math.min(rows, cols - k) : Math.min(rows + k, cols),
      Math.max(rows, cols),
    );
    if (len <= 0) return zeros([0], arr.dtype);
    const out = createTypedArray(arr.dtype, len);
    for (let i = 0; i < len; i++) {
      const row = k >= 0 ? i : i - k;
      const col = k >= 0 ? i + k : i;
      out[i] = arr.data[row * cols + col];
    }
    return new NdArray(out, [len], arr.dtype);
  }

  throw new Error("Input must be 1-D or 2-D");
};

export const zerosLike = (a: NdArray, dtype?: DtypeString): NdArray =>
  zeros(a.shape, dtype || a.dtype);
export const onesLike = (a: NdArray, dtype?: DtypeString): NdArray =>
  ones(a.shape, dtype || a.dtype);
export const fullLike = (
  a: NdArray,
  value: number,
  dtype?: DtypeString,
): NdArray => full(a.shape, value, dtype || a.dtype);
export const emptyLike = (a: NdArray, dtype?: DtypeString): NdArray =>
  empty(a.shape, dtype || a.dtype);

export const fromfunction = (
  fn: (...idx: number[]) => number,
  shape: Shape,
  dtype: DtypeString = "float64",
): NdArray => {
  const size = shapeSize(shape);
  const data = createTypedArray(dtype, size);
  for (let i = 0; i < size; i++) data[i] = fn(...flatToMulti(i, shape));
  return new NdArray(data, shape, dtype);
};

export const frombuffer = (
  buffer: ArrayBuffer,
  dtype: DtypeString = "float64",
  count = -1,
  offset = 0,
): NdArray => {
  const info = getDtypeInfo(dtype);
  const byteLen =
    count === -1 ? buffer.byteLength - offset : count * info.bytes;
  const length = byteLen / info.bytes;
  const typed = new info.ArrayType(buffer, offset, length) as any;
  return new NdArray(typed, [length], dtype);
};

export const tile = (A: NdArray, reps: number | number[]): NdArray => {
  const rep = typeof reps === "number" ? [reps] : reps;
  const ndim = Math.max(A.ndim, rep.length);

  const padShape = new Array(ndim).fill(1);
  const padReps = new Array(ndim).fill(1);
  for (let i = 0; i < A.ndim; i++) padShape[ndim - A.ndim + i] = A.shape[i];
  for (let i = 0; i < rep.length; i++) padReps[ndim - rep.length + i] = rep[i];

  const newShape = padShape.map((s, i) => s * padReps[i]);
  const size = shapeSize(newShape);
  const data = createTypedArray(A.dtype, size);
  const srcStrides = computeStrides(padShape);

  for (let i = 0; i < size; i++) {
    const idx = flatToMulti(i, newShape);
    const srcIdx = idx.map((v, d) => v % padShape[d]);
    data[i] = A.data[multiToFlat(srcIdx, srcStrides)] ?? 0;
  }
  return new NdArray(data, newShape, A.dtype);
};
