/**
 * Linear algebra — det, inv, solve, eig, svd, qr, cholesky, norm, etc.
 */

import { NdArray } from "../core/ndarray";
import { createTypedArray } from "../core/dtype";
import { eye } from "../creation";

// --- Core operations ---

export const dot = (a: NdArray, b: NdArray): NdArray => a.dot(b);

export const matmul = (a: NdArray, b: NdArray): NdArray =>
  a.ndim < 2 || b.ndim < 2 ? dot(a, b) : a.dot(b);

export const det = (a: NdArray): number => {
  if (a.ndim !== 2 || a.shape[0] !== a.shape[1])
    throw new Error("Expected square matrix");
  const n = a.shape[0];
  if (n === 1) return a.data[0];
  if (n === 2) return a.data[0] * a.data[3] - a.data[1] * a.data[2];

  // LU with partial pivoting
  const lu = new Float64Array(a.data);
  let sign = 1;

  for (let i = 0; i < n; i++) {
    let maxVal = Math.abs(lu[i * n + i]),
      maxRow = i;
    for (let k = i + 1; k < n; k++)
      if (Math.abs(lu[k * n + i]) > maxVal) {
        maxVal = Math.abs(lu[k * n + i]);
        maxRow = k;
      }

    if (maxRow !== i) {
      for (let j = 0; j < n; j++) {
        const tmp = lu[i * n + j];
        lu[i * n + j] = lu[maxRow * n + j];
        lu[maxRow * n + j] = tmp;
      }
      sign *= -1;
    }
    if (Math.abs(lu[i * n + i]) < 1e-15) return 0;

    for (let k = i + 1; k < n; k++) {
      lu[k * n + i] /= lu[i * n + i];
      for (let j = i + 1; j < n; j++)
        lu[k * n + j] -= lu[k * n + i] * lu[i * n + j];
    }
  }

  let result = sign;
  for (let i = 0; i < n; i++) result *= lu[i * n + i];
  return result;
};

export const inv = (a: NdArray): NdArray => {
  if (a.ndim !== 2 || a.shape[0] !== a.shape[1])
    throw new Error("Expected square matrix");
  const n = a.shape[0];
  const aug = new Float64Array(n * 2 * n);

  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) aug[i * 2 * n + j] = a.data[i * n + j];
    aug[i * 2 * n + n + i] = 1;
  }

  for (let i = 0; i < n; i++) {
    let maxVal = Math.abs(aug[i * 2 * n + i]),
      maxRow = i;
    for (let k = i + 1; k < n; k++)
      if (Math.abs(aug[k * 2 * n + i]) > maxVal) {
        maxVal = Math.abs(aug[k * 2 * n + i]);
        maxRow = k;
      }

    if (Math.abs(aug[maxRow * 2 * n + i]) < 1e-15)
      throw new Error("Singular matrix");

    if (maxRow !== i)
      for (let j = 0; j < 2 * n; j++) {
        const tmp = aug[i * 2 * n + j];
        aug[i * 2 * n + j] = aug[maxRow * 2 * n + j];
        aug[maxRow * 2 * n + j] = tmp;
      }

    const pivot = aug[i * 2 * n + i];
    for (let j = 0; j < 2 * n; j++) aug[i * 2 * n + j] /= pivot;

    for (let k = 0; k < n; k++) {
      if (k === i) continue;
      const factor = aug[k * 2 * n + i];
      for (let j = 0; j < 2 * n; j++)
        aug[k * 2 * n + j] -= factor * aug[i * 2 * n + j];
    }
  }

  const out = createTypedArray("float64", n * n);
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++) out[i * n + j] = aug[i * 2 * n + n + j];
  return new NdArray(out, [n, n], "float64");
};

export const solve = (A: NdArray, b: NdArray): NdArray => {
  if (A.ndim !== 2 || A.shape[0] !== A.shape[1])
    throw new Error("Expected square matrix");
  const n = A.shape[0];
  const aug = new Float64Array(n * (n + 1));

  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) aug[i * (n + 1) + j] = A.data[i * n + j];
    aug[i * (n + 1) + n] = b.data[i];
  }

  for (let i = 0; i < n; i++) {
    let maxVal = Math.abs(aug[i * (n + 1) + i]),
      maxRow = i;
    for (let k = i + 1; k < n; k++)
      if (Math.abs(aug[k * (n + 1) + i]) > maxVal) {
        maxVal = Math.abs(aug[k * (n + 1) + i]);
        maxRow = k;
      }

    if (Math.abs(aug[maxRow * (n + 1) + i]) < 1e-15)
      throw new Error("Singular matrix");

    if (maxRow !== i)
      for (let j = 0; j <= n; j++) {
        const tmp = aug[i * (n + 1) + j];
        aug[i * (n + 1) + j] = aug[maxRow * (n + 1) + j];
        aug[maxRow * (n + 1) + j] = tmp;
      }

    for (let k = i + 1; k < n; k++) {
      const factor = aug[k * (n + 1) + i] / aug[i * (n + 1) + i];
      for (let j = i; j <= n; j++)
        aug[k * (n + 1) + j] -= factor * aug[i * (n + 1) + j];
    }
  }

  const out = createTypedArray("float64", n);
  for (let i = n - 1; i >= 0; i--) {
    out[i] = aug[i * (n + 1) + n];
    for (let j = i + 1; j < n; j++) out[i] -= aug[i * (n + 1) + j] * out[j];
    out[i] /= aug[i * (n + 1) + i];
  }
  return new NdArray(out, [n], "float64");
};

// --- QR decomposition (Gram-Schmidt) ---

const qrDecomp = (
  A: Float64Array,
  n: number,
): { q: Float64Array; r: Float64Array } => {
  const Q = new Float64Array(n * n);
  const R = new Float64Array(n * n);

  // Extract columns
  const cols: Float64Array[] = [];
  for (let j = 0; j < n; j++) {
    const col = new Float64Array(n);
    for (let i = 0; i < n; i++) col[i] = A[i * n + j];
    cols.push(col);
  }

  const ortho: Float64Array[] = [];
  for (let j = 0; j < n; j++) {
    const v = new Float64Array(cols[j]);
    for (let k = 0; k < j; k++) {
      let d = 0,
        ns = 0;
      for (let i = 0; i < n; i++) {
        d += v[i] * ortho[k][i];
        ns += ortho[k][i] ** 2;
      }
      if (ns > 1e-15) {
        const p = d / ns;
        for (let i = 0; i < n; i++) v[i] -= p * ortho[k][i];
      }
    }
    ortho.push(v);

    let nm = 0;
    for (let i = 0; i < n; i++) nm += v[i] ** 2;
    nm = Math.sqrt(nm);
    if (nm > 1e-15) for (let i = 0; i < n; i++) Q[i * n + j] = v[i] / nm;
  }

  // R = Q^T A
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++) {
      let s = 0;
      for (let k = 0; k < n; k++) s += Q[k * n + i] * A[k * n + j];
      R[i * n + j] = s;
    }

  return { q: Q, r: R };
};

export const eig = (a: NdArray): { values: NdArray; vectors: NdArray } => {
  if (a.ndim !== 2 || a.shape[0] !== a.shape[1])
    throw new Error("Expected square matrix");
  const n = a.shape[0];

  let A = new Float64Array(a.data);
  let Qt = new Float64Array(n * n);
  for (let i = 0; i < n; i++) Qt[i * n + i] = 1;

  for (let iter = 0; iter < 100; iter++) {
    const { q, r } = qrDecomp(A, n);

    // A = R * Q
    const newA = new Float64Array(n * n);
    for (let i = 0; i < n; i++)
      for (let j = 0; j < n; j++) {
        let s = 0;
        for (let k = 0; k < n; k++) s += r[i * n + k] * q[k * n + j];
        newA[i * n + j] = s;
      }

    // Accumulate: Qt = Qt * Q
    const newQt = new Float64Array(n * n);
    for (let i = 0; i < n; i++)
      for (let j = 0; j < n; j++) {
        let s = 0;
        for (let k = 0; k < n; k++) s += Qt[i * n + k] * q[k * n + j];
        newQt[i * n + j] = s;
      }

    A = newA;
    Qt = newQt;

    let off = 0;
    for (let i = 0; i < n; i++)
      for (let j = 0; j < n; j++) if (i !== j) off += Math.abs(A[i * n + j]);
    if (off < 1e-10) break;
  }

  const vals = createTypedArray("float64", n);
  for (let i = 0; i < n; i++) vals[i] = A[i * n + i];

  const vecs = createTypedArray("float64", n * n);
  vecs.set(Qt);

  return {
    values: new NdArray(vals, [n], "float64"),
    vectors: new NdArray(vecs, [n, n], "float64"),
  };
};

export const svd = (a: NdArray): { u: NdArray; s: NdArray; vt: NdArray } => {
  if (a.ndim !== 2) throw new Error("Expected 2-D array");
  const [m, n] = a.shape;

  // A^T A
  const ata = createTypedArray("float64", n * n);
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++) {
      let s = 0;
      for (let k = 0; k < m; k++) s += a.data[k * n + i] * a.data[k * n + j];
      ata[i * n + j] = s;
    }

  const { values: eigVals, vectors: V } = eig(
    new NdArray(ata, [n, n], "float64"),
  );

  // Sort descending
  const order = Array.from({ length: n }, (_, i) => i).sort(
    (x, y) => eigVals.data[y] - eigVals.data[x],
  );
  const sortedEig = createTypedArray("float64", n);
  const sortedV = createTypedArray("float64", n * n);
  for (let i = 0; i < n; i++) {
    sortedEig[i] = eigVals.data[order[i]];
    for (let j = 0; j < n; j++) sortedV[j * n + i] = V.data[j * n + order[i]];
  }

  const k = Math.min(m, n);
  const s = createTypedArray("float64", k);
  for (let i = 0; i < k; i++) s[i] = Math.sqrt(Math.max(0, sortedEig[i]));

  // U = A V S^-1
  const u = createTypedArray("float64", m * m);
  for (let i = 0; i < m; i++)
    for (let j = 0; j < k; j++) {
      if (s[j] > 1e-15) {
        let sum = 0;
        for (let l = 0; l < n; l++)
          sum += a.data[i * n + l] * sortedV[l * n + j];
        u[i * m + j] = sum / s[j];
      }
    }
  for (let j = k; j < m; j++) u[j * m + j] = 1;

  const vt = createTypedArray("float64", n * n);
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++) vt[i * n + j] = sortedV[j * n + i];

  return {
    u: new NdArray(u, [m, m], "float64"),
    s: new NdArray(s, [k], "float64"),
    vt: new NdArray(vt, [n, n], "float64"),
  };
};

export const norm = (a: NdArray, ord?: number | "fro" | string): number => {
  if (a.ndim === 1) {
    if (ord === undefined || ord === 2) {
      let s = 0;
      for (let i = 0; i < a.size; i++) s += a.data[i] ** 2;
      return Math.sqrt(s);
    }
    if (ord === 1) {
      let s = 0;
      for (let i = 0; i < a.size; i++) s += Math.abs(a.data[i]);
      return s;
    }
    if (ord === Infinity) {
      let mx = 0;
      for (let i = 0; i < a.size; i++) mx = Math.max(mx, Math.abs(a.data[i]));
      return mx;
    }
    if (ord === -Infinity) {
      let mn = Infinity;
      for (let i = 0; i < a.size; i++) mn = Math.min(mn, Math.abs(a.data[i]));
      return mn;
    }
    if (typeof ord === "number") {
      let s = 0;
      for (let i = 0; i < a.size; i++) s += Math.abs(a.data[i]) ** ord;
      return s ** (1 / ord);
    }
  }
  // Frobenius for matrices / default
  let s = 0;
  for (let i = 0; i < a.size; i++) s += a.data[i] ** 2;
  return Math.sqrt(s);
};

export const qr = (a: NdArray): { Q: NdArray; R: NdArray } => {
  if (a.ndim !== 2) throw new Error("Expected 2-D array");
  const n = Math.min(a.shape[0], a.shape[1]);
  const { q, r } = qrDecomp(new Float64Array(a.data), n);
  const Q = createTypedArray("float64", a.shape[0] * n);
  const R = createTypedArray("float64", n * n);
  for (let i = 0; i < q.length && i < Q.length; i++) Q[i] = q[i];
  for (let i = 0; i < r.length && i < R.length; i++) R[i] = r[i];
  return {
    Q: new NdArray(Q, [a.shape[0], n], "float64"),
    R: new NdArray(R, [n, n], "float64"),
  };
};

export const cholesky = (a: NdArray): NdArray => {
  if (a.ndim !== 2 || a.shape[0] !== a.shape[1])
    throw new Error("Expected square matrix");
  const n = a.shape[0];
  const L = createTypedArray("float64", n * n);

  for (let i = 0; i < n; i++)
    for (let j = 0; j <= i; j++) {
      let s = 0;
      for (let k = 0; k < j; k++) s += L[i * n + k] * L[j * n + k];
      if (i === j) {
        const val = a.data[i * n + i] - s;
        if (val < 0) throw new Error("Not positive definite");
        L[i * n + j] = Math.sqrt(val);
      } else {
        L[i * n + j] = (a.data[i * n + j] - s) / L[j * n + j];
      }
    }
  return new NdArray(L, [n, n], "float64");
};

export const matrix_rank = (a: NdArray, tol?: number): number => {
  const { s } = svd(a);
  const threshold = tol ?? Math.max(...a.shape) * s.data[0] * 2.2e-16;
  let rank = 0;
  for (let i = 0; i < s.size; i++) if (s.data[i] > threshold) rank++;
  return rank;
};

export const trace = (a: NdArray, offset = 0): number => {
  if (a.ndim !== 2) throw new Error("Expected 2-D array");
  const [rows, cols] = a.shape;
  let s = 0;
  for (let i = 0; i < Math.min(rows, cols); i++) {
    const col = i + offset;
    if (col >= 0 && col < cols) s += a.data[i * cols + col];
  }
  return s;
};

export const matrix_power = (a: NdArray, n: number): NdArray => {
  if (a.ndim !== 2 || a.shape[0] !== a.shape[1])
    throw new Error("Expected square matrix");
  if (n === 0) return eye(a.shape[0]);

  let base = n < 0 ? inv(a) : a.copy();
  let exp = Math.abs(n);
  let result = eye(a.shape[0]);

  while (exp > 0) {
    if (exp % 2 === 1) result = result.dot(base);
    base = base.dot(base);
    exp = Math.floor(exp / 2);
  }
  return result;
};

export const lstsq = (a: NdArray, b: NdArray) => {
  const at = a.transpose();
  const ata = at.dot(a);
  const atb = at.dot(b);
  const x = solve(ata, atb);
  const predicted = a.dot(x);
  let residualSum = 0;
  for (let i = 0; i < b.size; i++)
    residualSum += (b.data[i] - predicted.data[i]) ** 2;
  const { s: sVals } = svd(a);
  const resData = createTypedArray("float64", 1);
  resData[0] = residualSum;
  return {
    x,
    residuals: new NdArray(resData, [1], "float64"),
    rank: matrix_rank(a),
    s: sVals,
  };
};

// --- Namespace export ---

export const linalg = {
  dot,
  matmul,
  det,
  inv,
  solve,
  eig,
  svd,
  norm,
  qr,
  cholesky,
  matrix_rank,
  trace,
  matrix_power,
  lstsq,
};
