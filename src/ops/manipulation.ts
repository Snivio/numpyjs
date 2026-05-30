/**
 * Array manipulation — reshape, concat, stack, where, unique, etc.
 */

import { NdArray } from "../core/ndarray";
import { createTypedArray } from "../core/dtype";
import {
  shapeSize,
  computeStrides,
  flatToMulti,
  multiToFlat,
} from "../core/shape";

export const concatenate = (arrays: NdArray[], axis = 0): NdArray => {
  if (arrays.length === 0) throw new Error("Need at least one array");
  const first = arrays[0];
  const ndim = first.ndim;
  const ax = axis < 0 ? axis + ndim : axis;
  if (ax < 0 || ax >= ndim) throw new Error("Invalid axis");

  let totalAxis = 0;
  for (const arr of arrays) {
    for (let d = 0; d < ndim; d++)
      if (d !== ax && arr.shape[d] !== first.shape[d])
        throw new Error("Shape mismatch");
    totalAxis += arr.shape[ax];
  }

  const outShape = [...first.shape];
  outShape[ax] = totalAxis;
  const out = createTypedArray("float64", shapeSize(outShape));
  const outStrides = computeStrides(outShape);
  let offset = 0;

  for (const arr of arrays) {
    const total = arr.size;
    for (let i = 0; i < total; i++) {
      const idx = flatToMulti(i, arr.shape);
      idx[ax] += offset;
      out[multiToFlat(idx, outStrides)] = arr.data[i];
    }
    offset += arr.shape[ax];
  }
  return new NdArray(out, outShape, "float64");
};

export const stack = (arrays: NdArray[], axis = 0): NdArray => {
  const expanded = arrays.map((a) => a.expandDims(axis));
  return concatenate(expanded, axis);
};

export const vstack = (arrays: NdArray[]): NdArray => {
  const fixed = arrays.map((a) => (a.ndim === 1 ? a.reshape([1, a.size]) : a));
  return concatenate(fixed, 0);
};

export const hstack = (arrays: NdArray[]): NdArray => {
  if (arrays[0].ndim === 1) return concatenate(arrays, 0);
  return concatenate(arrays, 1);
};

export const dstack = (arrays: NdArray[]): NdArray => {
  const fixed = arrays.map((a) => {
    if (a.ndim === 1) return a.reshape([1, a.size, 1]);
    if (a.ndim === 2) return a.reshape([a.shape[0], a.shape[1], 1]);
    return a;
  });
  return concatenate(fixed, 2);
};

export const split = (
  a: NdArray,
  indices: number | number[],
  axis = 0,
): NdArray[] => {
  const ax = axis < 0 ? axis + a.ndim : axis;
  const axisSize = a.shape[ax];
  let sections: number[];

  if (typeof indices === "number") {
    if (axisSize % indices !== 0) throw new Error("Cannot split evenly");
    const chunk = axisSize / indices;
    sections = Array.from({ length: indices }, (_, i) => i * chunk);
    sections.push(axisSize);
  } else {
    sections = [0, ...indices, axisSize];
  }

  const result: NdArray[] = [];
  for (let s = 0; s < sections.length - 1; s++) {
    const start = sections[s],
      end = sections[s + 1];
    const sliceShape = [...a.shape];
    sliceShape[ax] = end - start;
    const out = createTypedArray("float64", shapeSize(sliceShape));
    const srcStrides = computeStrides(a.shape);

    const total = shapeSize(sliceShape);
    for (let i = 0; i < total; i++) {
      const idx = flatToMulti(i, sliceShape);
      const srcIdx = [...idx];
      srcIdx[ax] += start;
      out[i] = a.data[multiToFlat(srcIdx, srcStrides)];
    }
    result.push(new NdArray(out, sliceShape, "float64"));
  }
  return result;
};

export const vsplit = (a: NdArray, indices: number | number[]): NdArray[] =>
  split(a, indices, 0);
export const hsplit = (a: NdArray, indices: number | number[]): NdArray[] =>
  split(a, indices, 1);

export const repeat = (a: NdArray, repeats: number, axis?: number): NdArray =>
  a.repeat(repeats, axis);

export const flip = (a: NdArray, axis?: number): NdArray => {
  const ax = axis ?? 0;
  const out = createTypedArray(a.dtype, a.size);
  const strides = computeStrides(a.shape);

  for (let i = 0; i < a.size; i++) {
    const idx = flatToMulti(i, a.shape);
    idx[ax] = a.shape[ax] - 1 - idx[ax];
    out[multiToFlat(idx, strides)] = a.data[i];
  }
  return new NdArray(out, [...a.shape], a.dtype);
};

export const fliplr = (a: NdArray): NdArray => flip(a, a.ndim - 1);
export const flipud = (a: NdArray): NdArray => flip(a, 0);

export const roll = (a: NdArray, shift: number, axis?: number): NdArray => {
  if (axis === undefined) {
    const flat = a.flatten();
    const n = flat.size;
    const s = ((shift % n) + n) % n;
    const out = createTypedArray(a.dtype, n);
    for (let i = 0; i < n; i++) out[(i + s) % n] = flat.data[i];
    return new NdArray(out, [...a.shape], a.dtype);
  }

  const ax = axis < 0 ? axis + a.ndim : axis;
  const n = a.shape[ax];
  const s = ((shift % n) + n) % n;
  const out = createTypedArray(a.dtype, a.size);
  const strides = computeStrides(a.shape);

  for (let i = 0; i < a.size; i++) {
    const idx = flatToMulti(i, a.shape);
    const newIdx = [...idx];
    newIdx[ax] = (idx[ax] + s) % n;
    out[multiToFlat(newIdx, strides)] = a.data[i];
  }
  return new NdArray(out, [...a.shape], a.dtype);
};

export const rot90 = (a: NdArray, k = 1): NdArray => {
  if (a.ndim !== 2) throw new Error("Expected 2-D array");
  k = ((k % 4) + 4) % 4;
  if (k === 0) return a.copy();
  const [rows, cols] = a.shape;

  if (k === 1) {
    const out = createTypedArray(a.dtype, a.size);
    for (let i = 0; i < rows; i++)
      for (let j = 0; j < cols; j++)
        out[j * rows + (rows - 1 - i)] = a.data[i * cols + j];
    return new NdArray(out, [cols, rows], a.dtype);
  }
  if (k === 2) {
    const out = createTypedArray(a.dtype, a.size);
    for (let i = 0; i < a.size; i++) out[a.size - 1 - i] = a.data[i];
    return new NdArray(out, [rows, cols], a.dtype);
  }
  // k === 3
  const out = createTypedArray(a.dtype, a.size);
  for (let i = 0; i < rows; i++)
    for (let j = 0; j < cols; j++)
      out[(cols - 1 - j) * rows + i] = a.data[i * cols + j];
  return new NdArray(out, [cols, rows], a.dtype);
};

export const pad = (
  a: NdArray,
  padWidth: number | [number, number],
  mode: "constant" | "edge" | "wrap" = "constant",
  constantValues = 0,
): NdArray => {
  if (a.ndim !== 1) throw new Error("pad only supports 1-D arrays currently");
  const [before, after] =
    typeof padWidth === "number" ? [padWidth, padWidth] : padWidth;
  const n = a.size;
  const out = createTypedArray(a.dtype, n + before + after);

  for (let i = 0; i < n; i++) out[before + i] = a.data[i];

  if (mode === "constant") {
    for (let i = 0; i < before; i++) out[i] = constantValues;
    for (let i = 0; i < after; i++) out[before + n + i] = constantValues;
  } else if (mode === "edge") {
    for (let i = 0; i < before; i++) out[i] = a.data[0];
    for (let i = 0; i < after; i++) out[before + n + i] = a.data[n - 1];
  } else {
    for (let i = 0; i < before; i++) out[before - 1 - i] = a.data[i % n];
    for (let i = 0; i < after; i++) out[before + n + i] = a.data[i % n];
  }
  return new NdArray(out, [n + before + after], a.dtype);
};

export const unique = (a: NdArray): NdArray => {
  const set = new Set(Array.from(a.data));
  const vals = [...set].sort((x, y) => x - y);
  const out = createTypedArray("float64", vals.length);
  for (let i = 0; i < vals.length; i++) out[i] = vals[i];
  return new NdArray(out, [vals.length], "float64");
};

export const where = (condition: NdArray, x: NdArray, y: NdArray): NdArray => {
  const out = createTypedArray("float64", condition.size);
  for (let i = 0; i < condition.size; i++)
    out[i] = condition.data[i] ? x.data[i] : y.data[i];
  return new NdArray(out, [...condition.shape], "float64");
};

export const nonzero = (a: NdArray): NdArray => {
  const indices: number[] = [];
  for (let i = 0; i < a.size; i++) if (a.data[i] !== 0) indices.push(i);
  const out = createTypedArray("int32", indices.length);
  for (let i = 0; i < indices.length; i++) out[i] = indices[i];
  return new NdArray(out, [indices.length], "int32");
};

export const argsort = (a: NdArray, axis?: number): NdArray => a.argsort(axis);

export const argmax = (a: NdArray, axis?: number): NdArray | number =>
  a.argmax(axis);

export const argmin = (a: NdArray, axis?: number): NdArray | number =>
  a.argmin(axis);

export const searchsorted = (a: NdArray, v: number): number => {
  let lo = 0,
    hi = a.size;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (a.data[mid] < v) lo = mid + 1;
    else hi = mid;
  }
  return lo;
};

export const outer = (a: NdArray, b: NdArray): NdArray => {
  const out = createTypedArray("float64", a.size * b.size);
  for (let i = 0; i < a.size; i++)
    for (let j = 0; j < b.size; j++)
      out[i * b.size + j] = a.data[i] * b.data[j];
  return new NdArray(out, [a.size, b.size], "float64");
};

export const inner = (a: NdArray, b: NdArray): number => {
  let s = 0;
  for (let i = 0; i < a.size; i++) s += a.data[i] * b.data[i];
  return s;
};

export const cross = (a: NdArray, b: NdArray): NdArray => {
  if (a.size !== 3 || b.size !== 3)
    throw new Error("cross product requires 3-element vectors");
  const out = createTypedArray("float64", 3);
  out[0] = a.data[1] * b.data[2] - a.data[2] * b.data[1];
  out[1] = a.data[2] * b.data[0] - a.data[0] * b.data[2];
  out[2] = a.data[0] * b.data[1] - a.data[1] * b.data[0];
  return new NdArray(out, [3], "float64");
};

export const kron = (a: NdArray, b: NdArray): NdArray => {
  if (a.ndim !== 2 || b.ndim !== 2) throw new Error("kron requires 2-D arrays");
  const [ar, ac] = a.shape,
    [br, bc] = b.shape;
  const out = createTypedArray("float64", ar * br * ac * bc);
  const outCols = ac * bc;
  for (let i = 0; i < ar; i++)
    for (let j = 0; j < ac; j++)
      for (let k = 0; k < br; k++)
        for (let l = 0; l < bc; l++)
          out[(i * br + k) * outCols + j * bc + l] =
            a.data[i * ac + j] * b.data[k * bc + l];
  return new NdArray(out, [ar * br, ac * bc], "float64");
};

export const diff = (a: NdArray, n = 1, _axis?: number): NdArray => {
  if (a.ndim !== 1) throw new Error("diff currently supports 1-D only");
  let current = Array.from(a.data);
  for (let iter = 0; iter < n; iter++) {
    const next: number[] = [];
    for (let i = 1; i < current.length; i++)
      next.push(current[i] - current[i - 1]);
    current = next;
  }
  const out = createTypedArray("float64", current.length);
  for (let i = 0; i < current.length; i++) out[i] = current[i];
  return new NdArray(out, [current.length], "float64");
};

export const gradient = (a: NdArray): NdArray => {
  if (a.ndim !== 1) throw new Error("gradient currently supports 1-D only");
  const n = a.size;
  const out = createTypedArray("float64", n);
  if (n === 1) {
    out[0] = 0;
    return new NdArray(out, [1], "float64");
  }
  out[0] = a.data[1] - a.data[0];
  out[n - 1] = a.data[n - 1] - a.data[n - 2];
  for (let i = 1; i < n - 1; i++) out[i] = (a.data[i + 1] - a.data[i - 1]) / 2;
  return new NdArray(out, [n], "float64");
};

export const cumsum = (a: NdArray): NdArray => {
  const out = createTypedArray("float64", a.size);
  out[0] = a.data[0];
  for (let i = 1; i < a.size; i++) out[i] = out[i - 1] + a.data[i];
  return new NdArray(out, [a.size], "float64");
};

export const cumprod = (a: NdArray): NdArray => {
  const out = createTypedArray("float64", a.size);
  out[0] = a.data[0];
  for (let i = 1; i < a.size; i++) out[i] = out[i - 1] * a.data[i];
  return new NdArray(out, [a.size], "float64");
};

export const meshgrid = (x: NdArray, y: NdArray): [NdArray, NdArray] => {
  const nx = x.size,
    ny = y.size;
  const X = createTypedArray("float64", ny * nx);
  const Y = createTypedArray("float64", ny * nx);
  for (let i = 0; i < ny; i++)
    for (let j = 0; j < nx; j++) {
      X[i * nx + j] = x.data[j];
      Y[i * nx + j] = y.data[i];
    }
  return [
    new NdArray(X, [ny, nx], "float64"),
    new NdArray(Y, [ny, nx], "float64"),
  ];
};
