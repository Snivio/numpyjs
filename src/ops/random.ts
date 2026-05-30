/**
 * Random number generation — seeded PRNG with various distributions
 */

import { NdArray } from "../core/ndarray";
import { createTypedArray } from "../core/dtype";
import { Shape, shapeSize } from "../core/shape";
import { cholesky } from "./linalg";
import { array as toArray } from "../creation";

// --- Xorshift128+ PRNG ---

class PRNG {
  private s0: number;
  private s1: number;

  constructor(seed?: number) {
    this.s0 = seed ?? (Math.random() * 2147483647) | 0;
    this.s1 = (this.s0 ^ 0x6d2b79f5) | 0;
    for (let i = 0; i < 20; i++) this.next(); // warm up
  }

  next = (): number => {
    let a = this.s0;
    const b = this.s1;
    this.s0 = b;
    a ^= a << 23;
    a ^= a >> 17;
    a ^= b;
    a ^= b >> 26;
    this.s1 = a;
    return ((this.s0 + this.s1) >>> 0) / 4294967296;
  };

  gaussian = (): number => {
    const u1 = this.next(),
      u2 = this.next();
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  };
}

let rng = new PRNG();

// --- Public API ---

export const seed = (s: number): void => {
  rng = new PRNG(s);
};

export const rand = (...shape: number[]): NdArray => {
  const s = shape.length === 0 ? [1] : shape;
  const size = shapeSize(s);
  const data = createTypedArray("float64", size);
  for (let i = 0; i < size; i++) data[i] = rng.next();
  return new NdArray(data, s, "float64");
};

export const randn = (...shape: number[]): NdArray => {
  const s = shape.length === 0 ? [1] : shape;
  const size = shapeSize(s);
  const data = createTypedArray("float64", size);
  for (let i = 0; i < size; i++) data[i] = rng.gaussian();
  return new NdArray(data, s, "float64");
};

export const randint = (
  low: number,
  high?: number,
  size?: Shape | number,
): NdArray => {
  if (high === undefined) {
    high = low;
    low = 0;
  }
  const shape =
    size === undefined ? [1] : typeof size === "number" ? [size] : size;
  const total = shapeSize(shape);
  const range = high - low;
  const data = createTypedArray("int32", total);
  for (let i = 0; i < total; i++)
    data[i] = Math.floor(rng.next() * range) + low;
  return new NdArray(data, shape, "int32");
};

export const uniform = (low = 0, high = 1, size?: Shape | number): NdArray => {
  const shape =
    size === undefined ? [1] : typeof size === "number" ? [size] : size;
  const total = shapeSize(shape);
  const data = createTypedArray("float64", total);
  for (let i = 0; i < total; i++) data[i] = low + rng.next() * (high - low);
  return new NdArray(data, shape, "float64");
};

export const normal = (loc = 0, scale = 1, size?: Shape | number): NdArray => {
  const shape =
    size === undefined ? [1] : typeof size === "number" ? [size] : size;
  const total = shapeSize(shape);
  const data = createTypedArray("float64", total);
  for (let i = 0; i < total; i++) data[i] = loc + scale * rng.gaussian();
  return new NdArray(data, shape, "float64");
};

export const binomial = (
  n: number,
  p: number,
  size?: Shape | number,
): NdArray => {
  const shape =
    size === undefined ? [1] : typeof size === "number" ? [size] : size;
  const total = shapeSize(shape);
  const data = createTypedArray("int32", total);
  for (let i = 0; i < total; i++) {
    let successes = 0;
    for (let j = 0; j < n; j++) if (rng.next() < p) successes++;
    data[i] = successes;
  }
  return new NdArray(data, shape, "int32");
};

export const poisson = (lam = 1, size?: Shape | number): NdArray => {
  const shape =
    size === undefined ? [1] : typeof size === "number" ? [size] : size;
  const total = shapeSize(shape);
  const data = createTypedArray("int32", total);
  const L = Math.exp(-lam);
  for (let i = 0; i < total; i++) {
    let k = 0,
      p = 1;
    do {
      k++;
      p *= rng.next();
    } while (p > L);
    data[i] = k - 1;
  }
  return new NdArray(data, shape, "int32");
};

export const exponential = (scale = 1, size?: Shape | number): NdArray => {
  const shape =
    size === undefined ? [1] : typeof size === "number" ? [size] : size;
  const total = shapeSize(shape);
  const data = createTypedArray("float64", total);
  for (let i = 0; i < total; i++) data[i] = -scale * Math.log(1 - rng.next());
  return new NdArray(data, shape, "float64");
};

// Gamma variate (Marsaglia & Tsang)
const gammaVariate = (alpha: number): number => {
  if (alpha < 1) return gammaVariate(alpha + 1) * rng.next() ** (1 / alpha);
  const d = alpha - 1 / 3,
    c = 1 / Math.sqrt(9 * d);
  while (true) {
    let x: number, v: number;
    do {
      x = rng.gaussian();
      v = 1 + c * x;
    } while (v <= 0);
    v = v ** 3;
    const u = rng.next();
    if (u < 1 - 0.0331 * x ** 4) return d * v;
    if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v;
  }
};

export const gamma = (
  shapeParam: number,
  scale = 1,
  size?: Shape | number,
): NdArray => {
  const shape =
    size === undefined ? [1] : typeof size === "number" ? [size] : size;
  const total = shapeSize(shape);
  const data = createTypedArray("float64", total);
  for (let i = 0; i < total; i++) data[i] = gammaVariate(shapeParam) * scale;
  return new NdArray(data, shape, "float64");
};

export const beta = (a: number, b: number, size?: Shape | number): NdArray => {
  const shape =
    size === undefined ? [1] : typeof size === "number" ? [size] : size;
  const total = shapeSize(shape);
  const data = createTypedArray("float64", total);
  for (let i = 0; i < total; i++) {
    const x = gammaVariate(a),
      y = gammaVariate(b);
    data[i] = x / (x + y);
  }
  return new NdArray(data, shape, "float64");
};

export const chisquare = (df: number, size?: Shape | number): NdArray =>
  gamma(df / 2, 2, size);

export const choice = (
  a: NdArray | number[] | number,
  size?: number,
  replace = true,
  p?: number[],
): NdArray => {
  const values =
    typeof a === "number"
      ? Array.from({ length: a }, (_, i) => i)
      : a instanceof NdArray
        ? Array.from(a.data)
        : a;
  const n = size || 1;
  const data = createTypedArray("float64", n);

  if (p) {
    const cumP = [...p];
    for (let i = 1; i < cumP.length; i++) cumP[i] += cumP[i - 1];
    for (let i = 0; i < n; i++) {
      const r = rng.next();
      let idx = 0;
      while (idx < cumP.length - 1 && r > cumP[idx]) idx++;
      data[i] = values[idx];
    }
  } else if (replace) {
    for (let i = 0; i < n; i++)
      data[i] = values[Math.floor(rng.next() * values.length)];
  } else {
    if (n > values.length)
      throw new Error("Cannot sample more than population without replacement");
    const pool = [...values];
    for (let i = 0; i < n; i++) {
      const idx = Math.floor(rng.next() * pool.length);
      data[i] = pool[idx];
      pool.splice(idx, 1);
    }
  }
  return new NdArray(data, [n], "float64");
};

export const shuffle = (a: NdArray): void => {
  for (let i = a.size - 1; i > 0; i--) {
    const j = Math.floor(rng.next() * (i + 1));
    const tmp = a.data[i];
    a.data[i] = a.data[j];
    a.data[j] = tmp;
  }
};

export const permutation = (x: number | NdArray): NdArray => {
  let arr: NdArray;
  if (typeof x === "number") {
    const data = createTypedArray("int32", x);
    for (let i = 0; i < x; i++) data[i] = i;
    arr = new NdArray(data, [x], "int32");
  } else {
    arr = x.copy();
  }
  shuffle(arr);
  return arr;
};

export const multivariate_normal = (
  meanVec: number[],
  covMat: number[][],
  size?: number,
): NdArray => {
  const n = meanVec.length;
  const numSamples = size || 1;
  const data = createTypedArray("float64", numSamples * n);
  const L = cholesky(toArray(covMat));

  for (let s = 0; s < numSamples; s++) {
    const z: number[] = [];
    for (let i = 0; i < n; i++) z.push(rng.gaussian());
    for (let i = 0; i < n; i++) {
      let val = meanVec[i];
      for (let j = 0; j <= i; j++) val += L.data[i * n + j] * z[j];
      data[s * n + i] = val;
    }
  }
  return new NdArray(data, numSamples === 1 ? [n] : [numSamples, n], "float64");
};

// --- Namespace export ---

export const random = {
  seed,
  rand,
  randn,
  randint,
  uniform,
  normal,
  binomial,
  poisson,
  exponential,
  beta,
  gamma,
  chisquare,
  choice,
  shuffle,
  permutation,
  multivariate_normal,
};
