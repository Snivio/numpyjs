/**
 * NdArray — the core N-dimensional array (like numpy.ndarray)
 */

import {
  DtypeString,
  TypedArray,
  createTypedArray,
  castTypedArray,
  getDtypeInfo,
} from "./dtype";
import {
  Shape,
  computeStrides,
  shapeSize,
  validateShape,
  flatToMulti,
  multiToFlat,
  shapesEqual,
} from "./shape";
import { BinaryBroadcastIterator } from "./broadcast";

export type NestedArray = number | boolean | NestedArray[];

export class NdArray {
  readonly data: TypedArray;
  readonly shape: Shape;
  readonly strides: number[];
  readonly dtype: DtypeString;
  readonly ndim: number;
  readonly size: number;

  constructor(data: TypedArray, shape: Shape, dtype: DtypeString) {
    validateShape(shape);
    this.data = data;
    this.shape = [...shape];
    this.strides = computeStrides(shape);
    this.dtype = dtype;
    this.ndim = shape.length;
    this.size = shapeSize(shape);
  }

  get nbytes(): number {
    return this.size * getDtypeInfo(this.dtype).bytes;
  }

  get T(): NdArray {
    return this.transpose();
  }

  get flat(): TypedArray {
    return this.data;
  }

  // --- Indexing ---

  get = (...indices: number[]): number => {
    if (indices.length !== this.ndim)
      throw new Error(`Expected ${this.ndim} indices, got ${indices.length}`);
    const resolved = indices.map((idx, i) => {
      const d = idx < 0 ? idx + this.shape[i] : idx;
      if (d < 0 || d >= this.shape[i])
        throw new Error(
          `Index ${idx} out of bounds for axis ${i} with size ${this.shape[i]}`,
        );
      return d;
    });
    return this.data[multiToFlat(resolved, this.strides)];
  };

  set = (value: number, ...indices: number[]): void => {
    if (indices.length !== this.ndim)
      throw new Error(`Expected ${this.ndim} indices, got ${indices.length}`);
    const resolved = indices.map((idx, i) => {
      const d = idx < 0 ? idx + this.shape[i] : idx;
      if (d < 0 || d >= this.shape[i])
        throw new Error(
          `Index ${idx} out of bounds for axis ${i} with size ${this.shape[i]}`,
        );
      return d;
    });
    this.data[multiToFlat(resolved, this.strides)] = value;
  };

  item = (index?: number): number => {
    if (index === undefined) {
      if (this.size !== 1)
        throw new Error("Can only convert size-1 array to scalar");
      return this.data[0];
    }
    return this.data[index < 0 ? index + this.size : index];
  };

  // --- Shape manipulation ---

  reshape = (newShape: Shape): NdArray => {
    const resolved = [...newShape];
    let inferIdx = -1;
    let known = 1;

    for (let i = 0; i < resolved.length; i++) {
      if (resolved[i] === -1) {
        if (inferIdx !== -1) throw new Error("Only one -1 dimension allowed");
        inferIdx = i;
      } else {
        known *= resolved[i];
      }
    }

    if (inferIdx !== -1) {
      if (this.size % known !== 0)
        throw new Error(`Cannot reshape size ${this.size} into (${newShape})`);
      resolved[inferIdx] = this.size / known;
    }

    if (shapeSize(resolved) !== this.size)
      throw new Error(`Cannot reshape size ${this.size} into (${resolved})`);

    return new NdArray(this.data, resolved, this.dtype);
  };

  transpose = (axes?: number[]): NdArray => {
    if (this.ndim < 2) return this.copy();

    const ax =
      axes ?? Array.from({ length: this.ndim }, (_, i) => this.ndim - 1 - i);
    if (ax.length !== this.ndim)
      throw new Error(`axes must have length ${this.ndim}`);

    const newShape = ax.map((a) => this.shape[a]);
    const newData = createTypedArray(this.dtype, this.size);
    const newStrides = computeStrides(newShape);

    for (let i = 0; i < this.size; i++) {
      const src = flatToMulti(i, this.shape);
      const dst = ax.map((a) => src[a]);
      newData[multiToFlat(dst, newStrides)] = this.data[i];
    }

    return new NdArray(newData, newShape, this.dtype);
  };

  flatten = (): NdArray =>
    new NdArray(this.data.slice(), [this.size], this.dtype);
  ravel = (): NdArray => this.flatten();
  copy = (): NdArray =>
    new NdArray(this.data.slice(), [...this.shape], this.dtype);

  astype = (dtype: DtypeString): NdArray =>
    dtype === this.dtype
      ? this.copy()
      : new NdArray(castTypedArray(this.data, dtype), [...this.shape], dtype);

  fill = (value: number): void => {
    this.data.fill(value);
  };

  squeeze = (axis?: number | number[]): NdArray => {
    const axes =
      axis !== undefined ? (Array.isArray(axis) ? axis : [axis]) : null;
    const newShape: number[] = [];

    for (let i = 0; i < this.shape.length; i++) {
      if (axes) {
        if (axes.includes(i)) {
          if (this.shape[i] !== 1)
            throw new Error(
              `Cannot squeeze axis ${i} with size ${this.shape[i]}`,
            );
        } else {
          newShape.push(this.shape[i]);
        }
      } else if (this.shape[i] !== 1) {
        newShape.push(this.shape[i]);
      }
    }

    if (newShape.length === 0) newShape.push(1);
    return new NdArray(this.data.slice(), newShape, this.dtype);
  };

  expandDims = (axis: number): NdArray => {
    const ax = axis < 0 ? this.ndim + 1 + axis : axis;
    if (ax < 0 || ax > this.ndim)
      throw new Error(`axis ${axis} out of bounds for ${this.ndim} dimensions`);
    const newShape = [...this.shape];
    newShape.splice(ax, 0, 1);
    return new NdArray(this.data.slice(), newShape, this.dtype);
  };

  repeat = (repeats: number, axis?: number): NdArray => {
    if (axis === undefined) {
      const newData = createTypedArray(this.dtype, this.size * repeats);
      for (let i = 0; i < this.size; i++)
        for (let r = 0; r < repeats; r++)
          newData[i * repeats + r] = this.data[i];
      return new NdArray(newData, [this.size * repeats], this.dtype);
    }

    const ax = axis < 0 ? axis + this.ndim : axis;
    const newShape = [...this.shape];
    newShape[ax] *= repeats;
    const newData = createTypedArray(this.dtype, shapeSize(newShape));
    const newStrides = computeStrides(newShape);

    for (let i = 0; i < this.size; i++) {
      const idx = flatToMulti(i, this.shape);
      for (let r = 0; r < repeats; r++) {
        const dst = [...idx];
        dst[ax] = idx[ax] * repeats + r;
        newData[multiToFlat(dst, newStrides)] = this.data[i];
      }
    }
    return new NdArray(newData, newShape, this.dtype);
  };

  swapaxes = (a1: number, a2: number): NdArray => {
    const axes = Array.from({ length: this.ndim }, (_, i) => i);
    axes[a1] = a2;
    axes[a2] = a1;
    return this.transpose(axes);
  };

  // --- Reductions ---

  sum = (axis?: number | null, keepdims = false): NdArray | number =>
    this._reduce((a, b) => a + b, 0, axis, keepdims);

  prod = (axis?: number | null, keepdims = false): NdArray | number =>
    this._reduce((a, b) => a * b, 1, axis, keepdims);

  mean = (axis?: number | null, keepdims = false): NdArray | number => {
    const s = this.sum(axis, keepdims);
    if (typeof s === "number") return s / this.size;
    const count =
      axis != null ? this.shape[axis < 0 ? axis + this.ndim : axis] : this.size;
    const result = createTypedArray(this.dtype, s.size);
    for (let i = 0; i < s.size; i++) result[i] = s.data[i] / count;
    return new NdArray(result, s.shape, this.dtype);
  };

  variance = (
    axis?: number | null,
    keepdims = false,
    ddof = 0,
  ): NdArray | number => {
    const m = this.mean(axis, true);
    if (typeof m === "number") {
      let acc = 0;
      for (let i = 0; i < this.size; i++) acc += (this.data[i] - m) ** 2;
      return acc / (this.size - ddof);
    }
    const mArr = m as NdArray;
    const iter = new BinaryBroadcastIterator(this.shape, mArr.shape);
    const diffs = createTypedArray("float64", iter.size);
    iter.forEach((out, iA, iB) => {
      diffs[out] = (this.data[iA] - mArr.data[iB]) ** 2;
    });
    const diffsArr = new NdArray(diffs, iter.shape, "float64");
    const result = diffsArr.sum(axis, keepdims);
    if (typeof result === "number") {
      const count =
        axis != null
          ? this.shape[axis < 0 ? axis + this.ndim : axis]
          : this.size;
      return result / (count - ddof);
    }
    const count =
      axis != null ? this.shape[axis < 0 ? axis + this.ndim : axis] : this.size;
    const out = createTypedArray("float64", result.size);
    for (let i = 0; i < result.size; i++)
      out[i] = result.data[i] / (count - ddof);
    return new NdArray(out, result.shape, "float64");
  };

  std = (
    axis?: number | null,
    keepdims = false,
    ddof = 0,
  ): NdArray | number => {
    const v = this.variance(axis, keepdims, ddof);
    if (typeof v === "number") return Math.sqrt(v);
    const out = createTypedArray("float64", (v as NdArray).size);
    for (let i = 0; i < out.length; i++)
      out[i] = Math.sqrt((v as NdArray).data[i]);
    return new NdArray(out, (v as NdArray).shape, "float64");
  };

  min = (axis?: number | null, keepdims = false): NdArray | number =>
    this._reduce((a, b) => Math.min(a, b), Infinity, axis, keepdims);

  max = (axis?: number | null, keepdims = false): NdArray | number =>
    this._reduce((a, b) => Math.max(a, b), -Infinity, axis, keepdims);

  argmin = (axis?: number | null): NdArray | number => {
    if (axis == null) {
      let min = Infinity,
        idx = 0;
      for (let i = 0; i < this.size; i++)
        if (this.data[i] < min) {
          min = this.data[i];
          idx = i;
        }
      return idx;
    }
    return this._argReduce(axis, (a, b) => a < b);
  };

  argmax = (axis?: number | null): NdArray | number => {
    if (axis == null) {
      let max = -Infinity,
        idx = 0;
      for (let i = 0; i < this.size; i++)
        if (this.data[i] > max) {
          max = this.data[i];
          idx = i;
        }
      return idx;
    }
    return this._argReduce(axis, (a, b) => a > b);
  };

  cumsum = (axis?: number | null): NdArray => {
    if (axis == null) {
      const out = createTypedArray(this.dtype, this.size);
      out[0] = this.data[0];
      for (let i = 1; i < this.size; i++) out[i] = out[i - 1] + this.data[i];
      return new NdArray(out, [this.size], this.dtype);
    }
    const ax = axis < 0 ? axis + this.ndim : axis;
    const result = this.copy();
    for (let i = 0; i < this.size; i++) {
      const idx = flatToMulti(i, this.shape);
      if (idx[ax] > 0) {
        const prev = [...idx];
        prev[ax]--;
        result.data[i] =
          result.data[multiToFlat(prev, this.strides)] + this.data[i];
      }
    }
    return result;
  };

  cumprod = (axis?: number | null): NdArray => {
    if (axis == null) {
      const out = createTypedArray(this.dtype, this.size);
      out[0] = this.data[0];
      for (let i = 1; i < this.size; i++) out[i] = out[i - 1] * this.data[i];
      return new NdArray(out, [this.size], this.dtype);
    }
    const ax = axis < 0 ? axis + this.ndim : axis;
    const result = this.copy();
    for (let i = 0; i < this.size; i++) {
      const idx = flatToMulti(i, this.shape);
      if (idx[ax] > 0) {
        const prev = [...idx];
        prev[ax]--;
        result.data[i] =
          result.data[multiToFlat(prev, this.strides)] * this.data[i];
      }
    }
    return result;
  };

  // --- Boolean tests ---

  all = (axis?: number | null): boolean | NdArray => {
    if (axis == null) {
      for (let i = 0; i < this.size; i++) if (this.data[i] === 0) return false;
      return true;
    }
    return this._reduce((a, b) => (b !== 0 ? a : 0), 1, axis, false) as NdArray;
  };

  any = (axis?: number | null): boolean | NdArray => {
    if (axis == null) {
      for (let i = 0; i < this.size; i++) if (this.data[i] !== 0) return true;
      return false;
    }
    return this._reduce((a, b) => (b !== 0 ? 1 : a), 0, axis, false) as NdArray;
  };

  // --- Sorting ---

  sort = (axis = -1): NdArray => {
    const ax = axis < 0 ? axis + this.ndim : axis;
    if (this.ndim === 1) {
      const sorted = this.data.slice();
      sorted.sort();
      return new NdArray(sorted, [...this.shape], this.dtype);
    }

    const result = this.copy();
    const axisSize = this.shape[ax];
    const numSlices = this.size / axisSize;

    for (let s = 0; s < numSlices; s++) {
      const baseIdx = this._sliceBaseIndex(s, ax);
      const values: number[] = [];
      for (let i = 0; i < axisSize; i++) {
        const idx = [...baseIdx];
        idx[ax] = i;
        values.push(result.data[multiToFlat(idx, this.strides)]);
      }
      values.sort((a, b) => a - b);
      for (let i = 0; i < axisSize; i++) {
        const idx = [...baseIdx];
        idx[ax] = i;
        result.data[multiToFlat(idx, this.strides)] = values[i];
      }
    }
    return result;
  };

  argsort = (axis = -1): NdArray => {
    const ax = axis < 0 ? axis + this.ndim : axis;
    if (this.ndim === 1) {
      const indices = Array.from({ length: this.size }, (_, i) => i);
      indices.sort((a, b) => this.data[a] - this.data[b]);
      const out = createTypedArray("int32", this.size);
      for (let i = 0; i < this.size; i++) out[i] = indices[i];
      return new NdArray(out, [...this.shape], "int32");
    }

    const result = createTypedArray("int32", this.size);
    const axisSize = this.shape[ax];
    const numSlices = this.size / axisSize;

    for (let s = 0; s < numSlices; s++) {
      const baseIdx = this._sliceBaseIndex(s, ax);
      const values: number[] = [];
      for (let i = 0; i < axisSize; i++) {
        const idx = [...baseIdx];
        idx[ax] = i;
        values.push(this.data[multiToFlat(idx, this.strides)]);
      }
      const indices = Array.from({ length: axisSize }, (_, i) => i);
      indices.sort((a, b) => values[a] - values[b]);
      for (let i = 0; i < axisSize; i++) {
        const idx = [...baseIdx];
        idx[ax] = i;
        result[multiToFlat(idx, this.strides)] = indices[i];
      }
    }
    return new NdArray(result, [...this.shape], "int32");
  };

  // --- Slicing ---

  slice = (
    ...slices: ([number, number, number?] | null | number)[]
  ): NdArray => {
    const resolved: [number, number, number][] = [];
    const isReduced: boolean[] = [];

    for (let i = 0; i < this.ndim; i++) {
      const s = i < slices.length ? slices[i] : null;
      if (s === null || s === undefined) {
        resolved.push([0, this.shape[i], 1]);
        isReduced.push(false);
      } else if (typeof s === "number") {
        const idx = s < 0 ? s + this.shape[i] : s;
        resolved.push([idx, idx + 1, 1]);
        isReduced.push(true);
      } else {
        const [start_, stop_, step = 1] = s;
        let start = start_,
          stop = stop_;
        if (start < 0) start += this.shape[i];
        if (stop < 0) stop += this.shape[i];
        start = Math.max(0, start);
        stop = Math.min(this.shape[i], stop);
        resolved.push([start, stop, step]);
        isReduced.push(false);
      }
    }

    const newShape = resolved
      .filter((_, i) => !isReduced[i])
      .map(([start, stop, step]) =>
        Math.max(0, Math.ceil((stop - start) / step)),
      );

    const newData = createTypedArray(this.dtype, shapeSize(newShape));
    let outIdx = 0;

    const iterate = (dim: number, multi: number[]): void => {
      if (dim === this.ndim) {
        newData[outIdx++] = this.data[multiToFlat(multi, this.strides)];
        return;
      }
      const [start, stop, step] = resolved[dim];
      if (isReduced[dim]) {
        multi[dim] = start;
        iterate(dim + 1, multi);
      } else {
        for (let i = start; i < stop; i += step) {
          multi[dim] = i;
          iterate(dim + 1, multi);
        }
      }
    };
    iterate(0, new Array(this.ndim));
    return new NdArray(newData, newShape, this.dtype);
  };

  // --- Element-wise ops ---

  add = (other: NdArray | number): NdArray =>
    this._binOp(other, (a, b) => a + b);
  subtract = (other: NdArray | number): NdArray =>
    this._binOp(other, (a, b) => a - b);
  multiply = (other: NdArray | number): NdArray =>
    this._binOp(other, (a, b) => a * b);
  divide = (other: NdArray | number): NdArray =>
    this._binOp(other, (a, b) => a / b);
  power = (other: NdArray | number): NdArray =>
    this._binOp(other, (a, b) => a ** b);
  mod = (other: NdArray | number): NdArray =>
    this._binOp(other, (a, b) => ((a % b) + b) % b);
  neg = (): NdArray => this._unaryOp((a) => -a);
  abs = (): NdArray => this._unaryOp(Math.abs);

  // --- Comparisons ---

  eq = (other: NdArray | number): NdArray =>
    this._binOp(other, (a, b) => (a === b ? 1 : 0), "bool");
  ne = (other: NdArray | number): NdArray =>
    this._binOp(other, (a, b) => (a !== b ? 1 : 0), "bool");
  gt = (other: NdArray | number): NdArray =>
    this._binOp(other, (a, b) => (a > b ? 1 : 0), "bool");
  gte = (other: NdArray | number): NdArray =>
    this._binOp(other, (a, b) => (a >= b ? 1 : 0), "bool");
  lt = (other: NdArray | number): NdArray =>
    this._binOp(other, (a, b) => (a < b ? 1 : 0), "bool");
  lte = (other: NdArray | number): NdArray =>
    this._binOp(other, (a, b) => (a <= b ? 1 : 0), "bool");

  // --- Dot / matmul ---

  dot = (other: NdArray): NdArray => {
    if (this.ndim === 1 && other.ndim === 1) {
      if (this.shape[0] !== other.shape[0])
        throw new Error(
          `Shapes not aligned: ${this.shape[0]} vs ${other.shape[0]}`,
        );
      let s = 0;
      for (let i = 0; i < this.shape[0]; i++) s += this.data[i] * other.data[i];
      const out = createTypedArray(this.dtype, 1);
      out[0] = s;
      return new NdArray(out, [], this.dtype);
    }
    if (this.ndim === 2 && other.ndim === 2) return this._matmul2d(other);
    if (this.ndim === 2 && other.ndim === 1) {
      if (this.shape[1] !== other.shape[0])
        throw new Error(
          `Shapes not aligned: ${this.shape[1]} vs ${other.shape[0]}`,
        );
      const out = createTypedArray(this.dtype, this.shape[0]);
      const K = this.shape[1];
      for (let i = 0; i < this.shape[0]; i++) {
        let s = 0;
        for (let k = 0; k < K; k++) s += this.data[i * K + k] * other.data[k];
        out[i] = s;
      }
      return new NdArray(out, [this.shape[0]], this.dtype);
    }
    throw new Error(
      `dot not supported for shapes (${this.shape}) and (${other.shape})`,
    );
  };

  // --- String repr ---

  toString = (): string =>
    this.ndim === 0 ? `${this.data[0]}` : this._fmtArr(0, 0);

  toJSON = () => ({
    data: Array.from(this.data),
    shape: this.shape,
    dtype: this.dtype,
  });

  tolist = (): NestedArray =>
    this.ndim === 0 ? this.data[0] : this._toNested(0, 0);

  // --- Private helpers ---

  private _matmul2d = (other: NdArray): NdArray => {
    if (this.shape[1] !== other.shape[0])
      throw new Error(`Shapes not aligned: (${this.shape}) @ (${other.shape})`);
    const [M, K] = this.shape;
    const N = other.shape[1];
    const out = createTypedArray(this.dtype, M * N);
    for (let i = 0; i < M; i++) {
      for (let k = 0; k < K; k++) {
        const aik = this.data[i * K + k];
        for (let j = 0; j < N; j++) {
          out[i * N + j] += aik * other.data[k * N + j];
        }
      }
    }
    return new NdArray(out, [M, N], this.dtype);
  };

  private _unaryOp = (
    fn: (a: number) => number,
    outDtype?: DtypeString,
  ): NdArray => {
    const dtype = outDtype || this.dtype;
    const out = createTypedArray(dtype, this.size);
    for (let i = 0; i < this.size; i++) out[i] = fn(this.data[i]);
    return new NdArray(out, [...this.shape], dtype);
  };

  private _binOp = (
    other: NdArray | number,
    fn: (a: number, b: number) => number,
    outDtype?: DtypeString,
  ): NdArray => {
    const dtype = outDtype || this.dtype;
    if (typeof other === "number") {
      const out = createTypedArray(dtype, this.size);
      for (let i = 0; i < this.size; i++) out[i] = fn(this.data[i], other);
      return new NdArray(out, [...this.shape], dtype);
    }
    if (shapesEqual(this.shape, other.shape)) {
      const out = createTypedArray(dtype, this.size);
      for (let i = 0; i < this.size; i++)
        out[i] = fn(this.data[i], other.data[i]);
      return new NdArray(out, [...this.shape], dtype);
    }
    const iter = new BinaryBroadcastIterator(this.shape, other.shape);
    const out = createTypedArray(dtype, iter.size);
    iter.forEach((oi, iA, iB) => {
      out[oi] = fn(this.data[iA], other.data[iB]);
    });
    return new NdArray(out, iter.shape, dtype);
  };

  private _reduce = (
    fn: (acc: number, val: number) => number,
    init: number,
    axis?: number | null,
    keepdims = false,
  ): NdArray | number => {
    if (axis == null) {
      let acc = init;
      for (let i = 0; i < this.size; i++) acc = fn(acc, this.data[i]);
      if (keepdims) {
        const out = createTypedArray(this.dtype, 1);
        out[0] = acc;
        return new NdArray(out, new Array(this.ndim).fill(1), this.dtype);
      }
      return acc;
    }

    const ax = axis < 0 ? axis + this.ndim : axis;
    if (ax < 0 || ax >= this.ndim)
      throw new Error(`axis ${axis} out of bounds`);

    const newShape: number[] = [];
    for (let i = 0; i < this.ndim; i++) {
      if (i !== ax) newShape.push(this.shape[i]);
      else if (keepdims) newShape.push(1);
    }

    const outSize = shapeSize(newShape);
    const out = createTypedArray(this.dtype, outSize);
    out.fill(init);
    const outStrides = computeStrides(newShape);

    for (let i = 0; i < this.size; i++) {
      const multi = flatToMulti(i, this.shape);
      const rIdx: number[] = [];
      for (let d = 0; d < this.ndim; d++) {
        if (d !== ax) rIdx.push(multi[d]);
        else if (keepdims) rIdx.push(0);
      }
      const flat = multiToFlat(rIdx, outStrides);
      out[flat] = fn(out[flat], this.data[i]);
    }
    return new NdArray(out, newShape, this.dtype);
  };

  private _argReduce = (
    axis: number,
    cmp: (a: number, b: number) => boolean,
  ): NdArray => {
    const ax = axis < 0 ? axis + this.ndim : axis;
    const newShape = this.shape.filter((_, i) => i !== ax);
    const outSize = shapeSize(newShape);
    const out = createTypedArray("int32", outSize);
    const best = new Float64Array(outSize).fill(
      cmp(1, 0) ? -Infinity : Infinity,
    );
    const outStrides = computeStrides(newShape);

    for (let i = 0; i < this.size; i++) {
      const multi = flatToMulti(i, this.shape);
      const rIdx = multi.filter((_, d) => d !== ax);
      const flat = multiToFlat(rIdx, outStrides);
      if (cmp(this.data[i], best[flat])) {
        best[flat] = this.data[i];
        out[flat] = multi[ax];
      }
    }
    return new NdArray(out, newShape, "int32");
  };

  private _sliceBaseIndex = (sliceIdx: number, axis: number): number[] => {
    const baseIdx = new Array(this.ndim).fill(0);
    let count = sliceIdx;
    for (let d = this.ndim - 1; d >= 0; d--) {
      if (d === axis) continue;
      baseIdx[d] = count % this.shape[d];
      count = (count - baseIdx[d]) / this.shape[d];
    }
    return baseIdx;
  };

  private _fmtArr = (dim: number, offset: number): string => {
    if (dim === this.ndim - 1) {
      const items: string[] = [];
      for (let i = 0; i < this.shape[dim]; i++)
        items.push(String(this.data[offset + i * this.strides[dim]]));
      return `[${items.join(", ")}]`;
    }
    const items: string[] = [];
    for (let i = 0; i < this.shape[dim]; i++)
      items.push(this._fmtArr(dim + 1, offset + i * this.strides[dim]));
    return `[${items.join(",\n ")}]`;
  };

  private _toNested = (dim: number, offset: number): NestedArray => {
    if (dim === this.ndim - 1) {
      const arr: number[] = [];
      for (let i = 0; i < this.shape[dim]; i++)
        arr.push(this.data[offset + i * this.strides[dim]]);
      return arr;
    }
    const arr: NestedArray[] = [];
    for (let i = 0; i < this.shape[dim]; i++)
      arr.push(this._toNested(dim + 1, offset + i * this.strides[dim]));
    return arr;
  };
}
