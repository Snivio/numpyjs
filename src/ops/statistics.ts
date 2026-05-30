/**
 * Statistical operations — mean, std, median, corrcoef, cov, etc.
 */

import { NdArray } from "../core/ndarray";
import { createTypedArray } from "../core/dtype";
import {
  shapeSize,
  flatToMulti,
  multiToFlat,
  computeStrides,
} from "../core/shape";
import { array } from "../creation";

export const sum = (
  a: NdArray,
  axis?: number | null,
  keepdims = false,
): NdArray | number => a.sum(axis, keepdims);

export const prod = (
  a: NdArray,
  axis?: number | null,
  keepdims = false,
): NdArray | number => a.prod(axis, keepdims);

export const mean = (
  a: NdArray,
  axis?: number | null,
  keepdims = false,
): NdArray | number => a.mean(axis, keepdims);

export const median = (a: NdArray, axis?: number | null): NdArray | number => {
  if (axis == null) {
    const sorted = Array.from(a.data).sort((x, y) => x - y);
    const n = sorted.length;
    return n % 2 === 0
      ? (sorted[n / 2 - 1] + sorted[n / 2]) / 2
      : sorted[n >> 1];
  }

  const ax = axis < 0 ? axis + a.ndim : axis;
  const axisSize = a.shape[ax];
  const newShape = a.shape.filter((_, i) => i !== ax);
  const outSize = shapeSize(newShape);
  const out = createTypedArray("float64", outSize);

  for (let i = 0; i < outSize; i++) {
    const rIdx = flatToMulti(i, newShape);
    const values: number[] = [];
    for (let j = 0; j < axisSize; j++) {
      const srcIdx: number[] = [];
      let ri = 0;
      for (let d = 0; d < a.ndim; d++) srcIdx.push(d === ax ? j : rIdx[ri++]);
      values.push(a.data[multiToFlat(srcIdx, computeStrides(a.shape))]);
    }
    values.sort((x, y) => x - y);
    const n = values.length;
    out[i] =
      n % 2 === 0 ? (values[n / 2 - 1] + values[n / 2]) / 2 : values[n >> 1];
  }
  return new NdArray(out, newShape, "float64");
};

export const variance = (
  a: NdArray,
  axis?: number | null,
  ddof = 0,
  keepdims = false,
): NdArray | number => a.variance(axis, keepdims, ddof);

export const var_ = variance;

export const std = (
  a: NdArray,
  axis?: number | null,
  ddof = 0,
  keepdims = false,
): NdArray | number => a.std(axis, keepdims, ddof);

export const amin = (
  a: NdArray,
  axis?: number | null,
  keepdims = false,
): NdArray | number => a.min(axis, keepdims);

export const amax = (
  a: NdArray,
  axis?: number | null,
  keepdims = false,
): NdArray | number => a.max(axis, keepdims);

export const ptp = (a: NdArray, axis?: number | null): NdArray | number => {
  const hi = a.max(axis);
  const lo = a.min(axis);
  if (typeof hi === "number" && typeof lo === "number") return hi - lo;
  return (hi as NdArray).subtract(lo as NdArray);
};

export const percentile = (a: NdArray, q: number): number => {
  if (q < 0 || q > 100) throw new Error("q must be in [0, 100]");
  const sorted = Array.from(a.data).sort((x, y) => x - y);
  const n = sorted.length;
  const idx = (q / 100) * (n - 1);
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  return lo === hi
    ? sorted[lo]
    : sorted[lo] * (1 - (idx - lo)) + sorted[hi] * (idx - lo);
};

export const quantile = (a: NdArray, q: number): number =>
  percentile(a, q * 100);

export const average = (a: NdArray, weights?: NdArray | number[]): number => {
  if (!weights) return mean(a) as number;
  const w = weights instanceof NdArray ? weights : array(weights);
  let sumWV = 0,
    sumW = 0;
  for (let i = 0; i < a.size; i++) {
    sumWV += a.data[i] * w.data[i];
    sumW += w.data[i];
  }
  return sumWV / sumW;
};

export const corrcoef = (x: NdArray, y?: NdArray): NdArray => {
  if (x.ndim === 1 && y) {
    const n = x.size;
    const mx = mean(x) as number,
      my = mean(y) as number;
    const sx = std(x) as number,
      sy = std(y) as number;
    let cov = 0;
    for (let i = 0; i < n; i++) cov += (x.data[i] - mx) * (y.data[i] - my);
    cov /= n;
    const r = cov / (sx * sy);
    const out = createTypedArray("float64", 4);
    out[0] = 1;
    out[1] = r;
    out[2] = r;
    out[3] = 1;
    return new NdArray(out, [2, 2], "float64");
  }

  if (x.ndim !== 2) throw new Error("corrcoef requires 1-D or 2-D input");
  const [nVars, nObs] = x.shape;
  const means: number[] = [],
    stds: number[] = [];

  for (let i = 0; i < nVars; i++) {
    let s = 0;
    for (let j = 0; j < nObs; j++) s += x.data[i * nObs + j];
    const m = s / nObs;
    means.push(m);
    let v = 0;
    for (let j = 0; j < nObs; j++) v += (x.data[i * nObs + j] - m) ** 2;
    stds.push(Math.sqrt(v / nObs));
  }

  const out = createTypedArray("float64", nVars * nVars);
  for (let i = 0; i < nVars; i++) {
    for (let j = 0; j < nVars; j++) {
      let c = 0;
      for (let k = 0; k < nObs; k++)
        c +=
          (x.data[i * nObs + k] - means[i]) * (x.data[j * nObs + k] - means[j]);
      out[i * nVars + j] = c / nObs / (stds[i] * stds[j]);
    }
  }
  return new NdArray(out, [nVars, nVars], "float64");
};

export const cov = (x: NdArray, y?: NdArray, ddof = 1): NdArray => {
  if (x.ndim === 1 && y) {
    const n = x.size;
    const mx = mean(x) as number,
      my = mean(y) as number;
    let cxx = 0,
      cxy = 0,
      cyy = 0;
    for (let i = 0; i < n; i++) {
      const dx = x.data[i] - mx,
        dy = y.data[i] - my;
      cxx += dx * dx;
      cxy += dx * dy;
      cyy += dy * dy;
    }
    const d = n - ddof;
    const out = createTypedArray("float64", 4);
    out[0] = cxx / d;
    out[1] = cxy / d;
    out[2] = cxy / d;
    out[3] = cyy / d;
    return new NdArray(out, [2, 2], "float64");
  }

  if (x.ndim !== 2) throw new Error("cov requires 1-D or 2-D input");
  const [nVars, nObs] = x.shape;
  const means: number[] = [];
  for (let i = 0; i < nVars; i++) {
    let s = 0;
    for (let j = 0; j < nObs; j++) s += x.data[i * nObs + j];
    means.push(s / nObs);
  }

  const d = nObs - ddof;
  const out = createTypedArray("float64", nVars * nVars);
  for (let i = 0; i < nVars; i++)
    for (let j = 0; j < nVars; j++) {
      let c = 0;
      for (let k = 0; k < nObs; k++)
        c +=
          (x.data[i * nObs + k] - means[i]) * (x.data[j * nObs + k] - means[j]);
      out[i * nVars + j] = c / d;
    }
  return new NdArray(out, [nVars, nVars], "float64");
};

export const histogram = (
  a: NdArray,
  bins = 10,
  range?: [number, number],
): { hist: NdArray; binEdges: NdArray } => {
  const data = a.data;
  const lo = range ? range[0] : Math.min(...Array.from(data));
  const hi = range ? range[1] : Math.max(...Array.from(data));
  const width = (hi - lo) / bins;

  const hist = createTypedArray("int32", bins);
  const edges = createTypedArray("float64", bins + 1);
  for (let i = 0; i <= bins; i++) edges[i] = lo + i * width;

  for (let i = 0; i < data.length; i++) {
    const v = data[i];
    if (v < lo || v > hi) continue;
    let bin = Math.floor((v - lo) / width);
    if (bin === bins) bin--;
    hist[bin]++;
  }

  return {
    hist: new NdArray(hist, [bins], "int32"),
    binEdges: new NdArray(edges, [bins + 1], "float64"),
  };
};
