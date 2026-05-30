/**
 * Shape and stride utilities for N-dimensional arrays
 */

export type Shape = number[];

export const computeStrides = (shape: Shape): number[] => {
  const n = shape.length;
  if (n === 0) return [];
  const strides = new Array(n);
  strides[n - 1] = 1;
  for (let i = n - 2; i >= 0; i--) strides[i] = strides[i + 1] * shape[i + 1];
  return strides;
};

export const shapeSize = (shape: Shape): number => {
  let size = 1;
  for (let i = 0; i < shape.length; i++) size *= shape[i];
  return size;
};

export const validateShape = (shape: Shape): void => {
  for (const dim of shape) {
    if (!Number.isInteger(dim) || dim < 0)
      throw new Error(`Invalid shape dimension: ${dim}`);
  }
};

export const flatToMulti = (flat: number, shape: Shape): number[] => {
  const n = shape.length;
  const idx = new Array(n);
  let rem = flat;
  for (let i = n - 1; i >= 0; i--) {
    idx[i] = rem % shape[i];
    rem = (rem - idx[i]) / shape[i];
  }
  return idx;
};

export const multiToFlat = (indices: number[], strides: number[]): number => {
  let idx = 0;
  for (let i = 0; i < indices.length; i++) idx += indices[i] * strides[i];
  return idx;
};

export const broadcastShapes = (...shapes: Shape[]): Shape => {
  if (shapes.length === 0) return [];
  const maxNdim = Math.max(...shapes.map((s) => s.length));
  const result = new Array(maxNdim).fill(1);

  for (const shape of shapes) {
    const offset = maxNdim - shape.length;
    for (let i = 0; i < shape.length; i++) {
      const dim = shape[i];
      const ri = i + offset;
      if (result[ri] === 1) result[ri] = dim;
      else if (dim !== 1 && dim !== result[ri])
        throw new Error(
          `Cannot broadcast shapes: ${shapes.map((s) => `(${s})`).join(" ")}`,
        );
    }
  }
  return result;
};

export const shapesEqual = (a: Shape, b: Shape): boolean => {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
};
