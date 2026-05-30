/**
 * numpyjs — NumPy-like tensor library for TypeScript
 */

export { NdArray } from "./core/ndarray";
export * from "./core/dtype";
export * from "./core/shape";
export { BroadcastIterator, BinaryBroadcastIterator } from "./core/broadcast";

// Creation routines
export {
  array,
  zeros,
  ones,
  full,
  empty,
  arange,
  linspace,
  logspace,
  eye,
  identity,
  diag,
  zerosLike,
  onesLike,
  fullLike,
  emptyLike,
  fromfunction,
  frombuffer,
  tile,
} from "./creation";

// Ops
export {
  add,
  subtract,
  multiply,
  divide,
  floorDivide,
  mod,
  power,
  remainder,
  negative,
  positive,
  absolute,
  abs,
  sign,
  sin,
  cos,
  tan,
  arcsin,
  arccos,
  arctan,
  arctan2,
  hypot,
  degrees,
  radians,
  sinh,
  cosh,
  tanh,
  arcsinh,
  arccosh,
  arctanh,
  exp,
  exp2,
  expm1,
  log,
  log2,
  log10,
  log1p,
  floor,
  ceil,
  round,
  trunc,
  fix,
  sqrt,
  cbrt,
  square,
  reciprocal,
  clip,
  maximum,
  minimum,
  equal,
  notEqual,
  greater,
  greaterEqual,
  less,
  lessEqual,
  logicalAnd,
  logicalOr,
  logicalXor,
  logicalNot,
  isnan,
  isinf,
  isfinite,
  pi,
  e,
  inf,
  nan,
  newaxis,
} from "./ops/math";

export {
  sum,
  prod,
  mean,
  median,
  variance,
  var_,
  std,
  amin,
  amax,
  ptp,
  percentile,
  quantile,
  average,
  corrcoef,
  cov,
  histogram,
} from "./ops/statistics";

export { linalg } from "./ops/linalg";
export { random } from "./ops/random";
export { fft } from "./ops";
export {
  concatenate,
  stack,
  vstack,
  hstack,
  dstack,
  split,
  vsplit,
  hsplit,
  repeat,
  flip,
  fliplr,
  flipud,
  roll,
  rot90,
  pad,
  unique,
  where,
  nonzero,
  argsort,
  argmax,
  argmin,
  searchsorted,
  outer,
  inner,
  cross,
  kron,
  diff,
  gradient,
  cumsum,
  cumprod,
  meshgrid,
} from "./ops/manipulation";

// Also export dot/matmul at top-level
export { dot, matmul } from "./ops/linalg";

// --- Default export: np namespace (snake_case API matching NumPy) ---

import { NdArray } from "./core/ndarray";
import * as creation from "./creation";
import * as math from "./ops/math";
import * as stats from "./ops/statistics";
import { linalg } from "./ops/linalg";
import { random } from "./ops/random";
import { fftModule as fft } from "./ops/fft";
import * as manipulation from "./ops/manipulation";
import { dot, matmul } from "./ops/linalg";

const np = {
  // Core
  NdArray,

  // Creation
  ...creation,

  // Math
  ...math,

  // Aliases to match NumPy snake_case
  floor_divide: math.floorDivide,
  arc_sin: math.arcsin,
  arc_cos: math.arccos,
  arc_tan: math.arctan,
  arc_tan2: math.arctan2,
  arc_sinh: math.arcsinh,
  arc_cosh: math.arccosh,
  arc_tanh: math.arctanh,
  not_equal: math.notEqual,
  greater_equal: math.greaterEqual,
  less_equal: math.lessEqual,
  logical_and: math.logicalAnd,
  logical_or: math.logicalOr,
  logical_xor: math.logicalXor,
  logical_not: math.logicalNot,

  // Stats
  ...stats,
  min: stats.amin,
  max: stats.amax,
  zeros_like: creation.zerosLike,
  ones_like: creation.onesLike,
  full_like: creation.fullLike,
  empty_like: creation.emptyLike,

  // Manipulation
  ...manipulation,

  // Linalg, Random, FFT
  linalg,
  random,
  fft,
  dot,
  matmul,
};

export default np;
