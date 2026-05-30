/**
 * Element-wise math operations — arithmetic, trig, exp, comparisons, etc.
 */

import { NdArray } from "../core/ndarray";
import { DtypeString, createTypedArray } from "../core/dtype";
import { BinaryBroadcastIterator } from "../core/broadcast";
import { shapesEqual } from "../core/shape";
import { array as toArray } from "../creation";

// --- Internal helpers ---

const unary = (
  a: NdArray,
  fn: (x: number) => number,
  dtype?: DtypeString,
): NdArray => {
  const d = dtype || a.dtype;
  const out = createTypedArray(d, a.size);
  for (let i = 0; i < a.size; i++) out[i] = fn(a.data[i]);
  return new NdArray(out, [...a.shape], d);
};

const binary = (
  a: NdArray | number,
  b: NdArray | number,
  fn: (x: number, y: number) => number,
  dtype?: DtypeString,
): NdArray => {
  const arrA = typeof a === "number" ? toArray([a]) : a;
  const arrB = typeof b === "number" ? toArray([b]) : b;
  const d = dtype || arrA.dtype;

  if (shapesEqual(arrA.shape, arrB.shape)) {
    const out = createTypedArray(d, arrA.size);
    for (let i = 0; i < arrA.size; i++) out[i] = fn(arrA.data[i], arrB.data[i]);
    return new NdArray(out, [...arrA.shape], d);
  }

  const iter = new BinaryBroadcastIterator(arrA.shape, arrB.shape);
  const out = createTypedArray(d, iter.size);
  iter.forEach((oi, iA, iB) => {
    out[oi] = fn(arrA.data[iA], arrB.data[iB]);
  });
  return new NdArray(out, iter.shape, d);
};

// --- Arithmetic ---

export const add = (a: NdArray | number, b: NdArray | number) =>
  binary(a, b, (x, y) => x + y);
export const subtract = (a: NdArray | number, b: NdArray | number) =>
  binary(a, b, (x, y) => x - y);
export const multiply = (a: NdArray | number, b: NdArray | number) =>
  binary(a, b, (x, y) => x * y);
export const divide = (a: NdArray | number, b: NdArray | number) =>
  binary(a, b, (x, y) => x / y);
export const floorDivide = (a: NdArray | number, b: NdArray | number) =>
  binary(a, b, (x, y) => Math.floor(x / y));
export const mod = (a: NdArray | number, b: NdArray | number) =>
  binary(a, b, (x, y) => ((x % y) + y) % y);
export const power = (a: NdArray | number, b: NdArray | number) =>
  binary(a, b, (x, y) => x ** y);
export const remainder = mod;

export const negative = (a: NdArray) => unary(a, (x) => -x);
export const positive = (a: NdArray) => a.copy();
export const absolute = (a: NdArray) => unary(a, Math.abs);
export const abs = absolute;
export const sign = (a: NdArray) =>
  unary(a, (x) => (x > 0 ? 1 : x < 0 ? -1 : 0));

// --- Trigonometric ---

export const sin = (a: NdArray) => unary(a, Math.sin);
export const cos = (a: NdArray) => unary(a, Math.cos);
export const tan = (a: NdArray) => unary(a, Math.tan);
export const arcsin = (a: NdArray) => unary(a, Math.asin);
export const arccos = (a: NdArray) => unary(a, Math.acos);
export const arctan = (a: NdArray) => unary(a, Math.atan);
export const arctan2 = (y: NdArray | number, x: NdArray | number) =>
  binary(y, x, Math.atan2);
export const hypot = (x: NdArray | number, y: NdArray | number) =>
  binary(x, y, Math.hypot);
export const degrees = (a: NdArray) => unary(a, (x) => (x * 180) / Math.PI);
export const radians = (a: NdArray) => unary(a, (x) => (x * Math.PI) / 180);

// --- Hyperbolic ---

export const sinh = (a: NdArray) => unary(a, Math.sinh);
export const cosh = (a: NdArray) => unary(a, Math.cosh);
export const tanh = (a: NdArray) => unary(a, Math.tanh);
export const arcsinh = (a: NdArray) => unary(a, Math.asinh);
export const arccosh = (a: NdArray) => unary(a, Math.acosh);
export const arctanh = (a: NdArray) => unary(a, Math.atanh);

// --- Exponential / Logarithmic ---

export const exp = (a: NdArray) => unary(a, Math.exp);
export const exp2 = (a: NdArray) => unary(a, (x) => 2 ** x);
export const expm1 = (a: NdArray) => unary(a, Math.expm1);
export const log = (a: NdArray) => unary(a, Math.log);
export const log2 = (a: NdArray) => unary(a, Math.log2);
export const log10 = (a: NdArray) => unary(a, Math.log10);
export const log1p = (a: NdArray) => unary(a, Math.log1p);

// --- Rounding ---

export const floor = (a: NdArray) => unary(a, Math.floor);
export const ceil = (a: NdArray) => unary(a, Math.ceil);
export const round = (a: NdArray, decimals = 0) => {
  const f = 10 ** decimals;
  return unary(a, (x) => Math.round(x * f) / f);
};
export const trunc = (a: NdArray) => unary(a, Math.trunc);
export const fix = trunc;

// --- Other math ---

export const sqrt = (a: NdArray) => unary(a, Math.sqrt);
export const cbrt = (a: NdArray) => unary(a, Math.cbrt);
export const square = (a: NdArray) => unary(a, (x) => x * x);
export const reciprocal = (a: NdArray) => unary(a, (x) => 1 / x);
export const clip = (a: NdArray, lo: number | null, hi: number | null) =>
  unary(a, (x) => {
    if (lo !== null && x < lo) return lo;
    if (hi !== null && x > hi) return hi;
    return x;
  });

// --- Comparison ---

export const maximum = (a: NdArray | number, b: NdArray | number) =>
  binary(a, b, Math.max);
export const minimum = (a: NdArray | number, b: NdArray | number) =>
  binary(a, b, Math.min);
export const equal = (a: NdArray | number, b: NdArray | number) =>
  binary(a, b, (x, y) => (x === y ? 1 : 0), "bool");
export const notEqual = (a: NdArray | number, b: NdArray | number) =>
  binary(a, b, (x, y) => (x !== y ? 1 : 0), "bool");
export const greater = (a: NdArray | number, b: NdArray | number) =>
  binary(a, b, (x, y) => (x > y ? 1 : 0), "bool");
export const greaterEqual = (a: NdArray | number, b: NdArray | number) =>
  binary(a, b, (x, y) => (x >= y ? 1 : 0), "bool");
export const less = (a: NdArray | number, b: NdArray | number) =>
  binary(a, b, (x, y) => (x < y ? 1 : 0), "bool");
export const lessEqual = (a: NdArray | number, b: NdArray | number) =>
  binary(a, b, (x, y) => (x <= y ? 1 : 0), "bool");

// --- Logical ---

export const logicalAnd = (a: NdArray | number, b: NdArray | number) =>
  binary(a, b, (x, y) => (x && y ? 1 : 0), "bool");
export const logicalOr = (a: NdArray | number, b: NdArray | number) =>
  binary(a, b, (x, y) => (x || y ? 1 : 0), "bool");
export const logicalXor = (a: NdArray | number, b: NdArray | number) =>
  binary(a, b, (x, y) => (x ? 1 : 0) ^ (y ? 1 : 0), "bool");
export const logicalNot = (a: NdArray) => unary(a, (x) => (x ? 0 : 1), "bool");

// --- Special value checks ---

export const isnan = (a: NdArray) =>
  unary(a, (x) => (Number.isNaN(x) ? 1 : 0), "bool");
export const isinf = (a: NdArray) =>
  unary(a, (x) => (!Number.isFinite(x) && !Number.isNaN(x) ? 1 : 0), "bool");
export const isfinite = (a: NdArray) =>
  unary(a, (x) => (Number.isFinite(x) ? 1 : 0), "bool");

// --- Constants ---

export const pi = Math.PI;
export const e = Math.E;
export const inf = Infinity;
export const nan = NaN;
export const newaxis = null;
