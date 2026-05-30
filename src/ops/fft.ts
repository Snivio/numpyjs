/**
 * FFT — Cooley-Tukey radix-2 DIT
 */

import { NdArray } from "../core/ndarray";
import { createTypedArray } from "../core/dtype";

// Internal complex FFT
const coreFFT = (
  reIn: Float64Array,
  imIn: Float64Array,
  inverse: boolean,
): { re: Float64Array; im: Float64Array } => {
  const n = reIn.length;

  // Zero-pad to next power of 2
  let logN = 0;
  let N = 1;
  while (N < n) {
    N <<= 1;
    logN++;
  }

  const re = new Float64Array(N);
  const im = new Float64Array(N);
  re.set(reIn);
  im.set(imIn);

  // Bit-reversal permutation
  for (let i = 0; i < N; i++) {
    let j = 0;
    for (let bit = 0; bit < logN; bit++)
      if (i & (1 << bit)) j |= 1 << (logN - 1 - bit);
    if (j > i) {
      [re[i], re[j]] = [re[j], re[i]];
      [im[i], im[j]] = [im[j], im[i]];
    }
  }

  // Butterfly passes
  const sign = inverse ? 1 : -1;
  for (let size = 2; size <= N; size <<= 1) {
    const half = size >> 1;
    const angle = (sign * 2 * Math.PI) / size;
    const wRe = Math.cos(angle),
      wIm = Math.sin(angle);

    for (let i = 0; i < N; i += size) {
      let tRe = 1,
        tIm = 0;
      for (let j = 0; j < half; j++) {
        const a = i + j,
          b = a + half;
        const uRe = re[a],
          uIm = im[a];
        const vRe = re[b] * tRe - im[b] * tIm;
        const vIm = re[b] * tIm + im[b] * tRe;
        re[a] = uRe + vRe;
        im[a] = uIm + vIm;
        re[b] = uRe - vRe;
        im[b] = uIm - vIm;
        const newTRe = tRe * wRe - tIm * wIm;
        tIm = tRe * wIm + tIm * wRe;
        tRe = newTRe;
      }
    }
  }

  if (inverse) {
    for (let i = 0; i < N; i++) {
      re[i] /= N;
      im[i] /= N;
    }
  }

  return { re, im };
};

export const fft = (a: NdArray): NdArray => {
  const reIn = new Float64Array(a.data);
  const imIn = new Float64Array(a.size);
  const { re, im } = coreFFT(reIn, imIn, false);

  // Interleave real & imag into a 2-col array [n, 2]
  const n = re.length;
  const out = createTypedArray("float64", n * 2);
  for (let i = 0; i < n; i++) {
    out[i * 2] = re[i];
    out[i * 2 + 1] = im[i];
  }
  return new NdArray(out, [n, 2], "float64");
};

export const ifft = (a: NdArray): NdArray => {
  let reIn: Float64Array, imIn: Float64Array;
  if (a.ndim === 2 && a.shape[1] === 2) {
    const n = a.shape[0];
    reIn = new Float64Array(n);
    imIn = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      reIn[i] = a.data[i * 2];
      imIn[i] = a.data[i * 2 + 1];
    }
  } else {
    reIn = new Float64Array(a.data);
    imIn = new Float64Array(a.size);
  }
  const { re, im } = coreFFT(reIn, imIn, true);
  const n = re.length;
  const out = createTypedArray("float64", n * 2);
  for (let i = 0; i < n; i++) {
    out[i * 2] = re[i];
    out[i * 2 + 1] = im[i];
  }
  return new NdArray(out, [n, 2], "float64");
};

export const fft2 = (a: NdArray): NdArray => {
  if (a.ndim !== 2) throw new Error("Expected 2-D array");
  const [rows, cols] = a.shape;

  // FFT each row
  const reData = new Float64Array(rows * cols);
  for (let i = 0; i < rows * cols; i++) reData[i] = a.data[i];

  // Pad cols to power of 2
  let N = 1;
  while (N < cols) N <<= 1;
  const rePadded = new Float64Array(rows * N);
  const imPadded = new Float64Array(rows * N);
  for (let r = 0; r < rows; r++)
    for (let c = 0; c < cols; c++) rePadded[r * N + c] = reData[r * cols + c];

  for (let r = 0; r < rows; r++) {
    const rowRe = rePadded.subarray(r * N, (r + 1) * N);
    const rowIm = imPadded.subarray(r * N, (r + 1) * N);
    const { re, im } = coreFFT(
      new Float64Array(rowRe),
      new Float64Array(rowIm),
      false,
    );
    rowRe.set(re);
    rowIm.set(im);
  }

  // FFT each column
  let M = 1;
  while (M < rows) M <<= 1;

  const colRe = new Float64Array(M);
  const colIm = new Float64Array(M);
  const outRe = new Float64Array(M * N);
  const outIm = new Float64Array(M * N);
  for (let r = 0; r < rows; r++)
    for (let c = 0; c < N; c++) {
      outRe[r * N + c] = rePadded[r * N + c];
      outIm[r * N + c] = imPadded[r * N + c];
    }

  for (let c = 0; c < N; c++) {
    for (let r = 0; r < M; r++) {
      colRe[r] = r < rows ? outRe[r * N + c] : 0;
      colIm[r] = r < rows ? outIm[r * N + c] : 0;
    }
    const { re, im } = coreFFT(
      new Float64Array(colRe),
      new Float64Array(colIm),
      false,
    );
    for (let r = 0; r < M; r++) {
      outRe[r * N + c] = re[r];
      outIm[r * N + c] = im[r];
    }
  }

  const total = M * N;
  const out = createTypedArray("float64", total * 2);
  for (let i = 0; i < total; i++) {
    out[i * 2] = outRe[i];
    out[i * 2 + 1] = outIm[i];
  }
  return new NdArray(out, [M, N, 2], "float64");
};

export const ifft2 = (a: NdArray): NdArray => {
  if (a.ndim !== 3 || a.shape[2] !== 2)
    throw new Error("Expected 3-D array with last dim 2 (complex)");
  const [rows, cols] = a.shape;
  const reData = new Float64Array(rows * cols);
  const imData = new Float64Array(rows * cols);
  for (let i = 0; i < rows * cols; i++) {
    reData[i] = a.data[i * 2];
    imData[i] = a.data[i * 2 + 1];
  }

  // IFFT rows
  for (let r = 0; r < rows; r++) {
    const rowRe = reData.subarray(r * cols, (r + 1) * cols);
    const rowIm = imData.subarray(r * cols, (r + 1) * cols);
    const { re, im } = coreFFT(
      new Float64Array(rowRe),
      new Float64Array(rowIm),
      true,
    );
    rowRe.set(re);
    rowIm.set(im);
  }

  // IFFT cols
  const colRe = new Float64Array(rows);
  const colIm = new Float64Array(rows);
  for (let c = 0; c < cols; c++) {
    for (let r = 0; r < rows; r++) {
      colRe[r] = reData[r * cols + c];
      colIm[r] = imData[r * cols + c];
    }
    const { re, im } = coreFFT(
      new Float64Array(colRe),
      new Float64Array(colIm),
      true,
    );
    for (let r = 0; r < rows; r++) {
      reData[r * cols + c] = re[r];
      imData[r * cols + c] = im[r];
    }
  }

  const out = createTypedArray("float64", rows * cols * 2);
  for (let i = 0; i < rows * cols; i++) {
    out[i * 2] = reData[i];
    out[i * 2 + 1] = imData[i];
  }
  return new NdArray(out, [rows, cols, 2], "float64");
};

export const fftfreq = (n: number, d = 1): NdArray => {
  const out = createTypedArray("float64", n);
  const val = 1 / (n * d);
  const half = Math.floor(n / 2);
  for (let i = 0; i <= half; i++) out[i] = i * val;
  for (let i = half + 1; i < n; i++) out[i] = (i - n) * val;
  return new NdArray(out, [n], "float64");
};

export const fftshift = (a: NdArray): NdArray => {
  const n = a.shape[0];
  const half = Math.floor(n / 2);
  const out = createTypedArray(a.dtype, a.size);
  const stride = a.size / n;
  for (let i = 0; i < n; i++) {
    const src = ((i + half) % n) * stride;
    const dst = i * stride;
    for (let j = 0; j < stride; j++) out[dst + j] = a.data[src + j];
  }
  return new NdArray(out, [...a.shape], a.dtype);
};

export const ifftshift = (a: NdArray): NdArray => {
  const n = a.shape[0];
  const half = Math.ceil(n / 2);
  const out = createTypedArray(a.dtype, a.size);
  const stride = a.size / n;
  for (let i = 0; i < n; i++) {
    const src = ((i + half) % n) * stride;
    const dst = i * stride;
    for (let j = 0; j < stride; j++) out[dst + j] = a.data[src + j];
  }
  return new NdArray(out, [...a.shape], a.dtype);
};

// --- Namespace export ---

export const fftModule = {
  fft,
  ifft,
  fft2,
  ifft2,
  fftfreq,
  fftshift,
  ifftshift,
};
