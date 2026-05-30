/**
 * Broadcasting iterators for element-wise operations on mismatched shapes
 */

import { Shape, computeStrides, shapeSize, broadcastShapes } from "./shape";

export class BroadcastIterator {
  readonly shape: Shape;
  readonly size: number;
  private readonly ndim: number;
  private readonly strides: number[][];

  constructor(shapes: Shape[]) {
    this.shape = broadcastShapes(...shapes);
    this.size = shapeSize(this.shape);
    this.ndim = this.shape.length;
    this.strides = shapes.map((s) => this.broadcastStrides(s));
  }

  private broadcastStrides = (shape: Shape): number[] => {
    const orig = computeStrides(shape);
    const offset = this.ndim - shape.length;
    const result = new Array(this.ndim).fill(0);
    for (let i = 0; i < shape.length; i++) {
      if (shape[i] !== 1) result[i + offset] = orig[i];
    }
    return result;
  };

  getIndices = (flat: number): number[] => {
    const multi = new Array(this.ndim);
    let rem = flat;
    for (let i = this.ndim - 1; i >= 0; i--) {
      multi[i] = rem % this.shape[i];
      rem = (rem - multi[i]) / this.shape[i];
    }
    return this.strides.map((s) => {
      let idx = 0;
      for (let d = 0; d < this.ndim; d++) idx += multi[d] * s[d];
      return idx;
    });
  };
}

export class BinaryBroadcastIterator {
  readonly shape: Shape;
  readonly size: number;
  private readonly ndim: number;
  private readonly stridesA: number[];
  private readonly stridesB: number[];

  constructor(shapeA: Shape, shapeB: Shape) {
    this.shape = broadcastShapes(shapeA, shapeB);
    this.size = shapeSize(this.shape);
    this.ndim = this.shape.length;
    this.stridesA = this.broadcastStrides(shapeA);
    this.stridesB = this.broadcastStrides(shapeB);
  }

  private broadcastStrides = (shape: Shape): number[] => {
    const orig = computeStrides(shape);
    const offset = this.ndim - shape.length;
    const result = new Array(this.ndim).fill(0);
    for (let i = 0; i < shape.length; i++) {
      if (shape[i] !== 1) result[i + offset] = orig[i];
    }
    return result;
  };

  forEach = (
    fn: (outIdx: number, idxA: number, idxB: number) => void,
  ): void => {
    const multi = new Array(this.ndim);
    for (let i = 0; i < this.size; i++) {
      let rem = i;
      for (let d = this.ndim - 1; d >= 0; d--) {
        multi[d] = rem % this.shape[d];
        rem = (rem - multi[d]) / this.shape[d];
      }
      let idxA = 0,
        idxB = 0;
      for (let d = 0; d < this.ndim; d++) {
        idxA += multi[d] * this.stridesA[d];
        idxB += multi[d] * this.stridesB[d];
      }
      fn(i, idxA, idxB);
    }
  };
}
