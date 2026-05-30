import { describe, it, expect } from "vitest";
import np, {
  array,
  zeros,
  ones,
  full,
  arange,
  linspace,
  eye,
  identity,
  diag,
  NdArray,
} from "../src";

describe("Array Creation", () => {
  it("should create array from nested data", () => {
    const a = array([1, 2, 3]);
    expect(a.shape).toEqual([3]);
    expect(a.size).toBe(3);
    expect(a.ndim).toBe(1);
    expect(Array.from(a.data)).toEqual([1, 2, 3]);
  });

  it("should create 2D array", () => {
    const a = array([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    expect(a.shape).toEqual([2, 3]);
    expect(a.size).toBe(6);
    expect(a.ndim).toBe(2);
    expect(a.get(0, 0)).toBe(1);
    expect(a.get(1, 2)).toBe(6);
  });

  it("should create 3D array", () => {
    const a = array([
      [
        [1, 2],
        [3, 4],
      ],
      [
        [5, 6],
        [7, 8],
      ],
    ]);
    expect(a.shape).toEqual([2, 2, 2]);
    expect(a.get(1, 1, 1)).toBe(8);
  });

  it("should create zeros", () => {
    const a = zeros([2, 3]);
    expect(a.shape).toEqual([2, 3]);
    expect(Array.from(a.data)).toEqual([0, 0, 0, 0, 0, 0]);
  });

  it("should create ones", () => {
    const a = ones([3, 2]);
    expect(a.shape).toEqual([3, 2]);
    for (let i = 0; i < a.size; i++) {
      expect(a.data[i]).toBe(1);
    }
  });

  it("should create full", () => {
    const a = full([2, 2], 7);
    expect(Array.from(a.data)).toEqual([7, 7, 7, 7]);
  });

  it("should create arange", () => {
    const a = arange(5);
    expect(Array.from(a.data)).toEqual([0, 1, 2, 3, 4]);

    const b = arange(2, 7);
    expect(Array.from(b.data)).toEqual([2, 3, 4, 5, 6]);

    const c = arange(0, 10, 2);
    expect(Array.from(c.data)).toEqual([0, 2, 4, 6, 8]);
  });

  it("should create linspace", () => {
    const a = linspace(0, 1, 5);
    expect(a.size).toBe(5);
    expect(a.data[0]).toBeCloseTo(0);
    expect(a.data[4]).toBeCloseTo(1);
    expect(a.data[2]).toBeCloseTo(0.5);
  });

  it("should create eye", () => {
    const a = eye(3);
    expect(a.get(0, 0)).toBe(1);
    expect(a.get(1, 1)).toBe(1);
    expect(a.get(2, 2)).toBe(1);
    expect(a.get(0, 1)).toBe(0);
  });

  it("should create identity", () => {
    const a = identity(4);
    expect(a.shape).toEqual([4, 4]);
    for (let i = 0; i < 4; i++) {
      expect(a.get(i, i)).toBe(1);
    }
  });

  it("should create/extract diagonal", () => {
    const d = diag(array([1, 2, 3]));
    expect(d.shape).toEqual([3, 3]);
    expect(d.get(0, 0)).toBe(1);
    expect(d.get(1, 1)).toBe(2);
    expect(d.get(2, 2)).toBe(3);
    expect(d.get(0, 1)).toBe(0);
  });
});

describe("NdArray operations", () => {
  it("should reshape", () => {
    const a = arange(12);
    const b = a.reshape([3, 4]);
    expect(b.shape).toEqual([3, 4]);
    expect(b.get(0, 0)).toBe(0);
    expect(b.get(2, 3)).toBe(11);
  });

  it("should reshape with -1", () => {
    const a = arange(12);
    const b = a.reshape([3, -1]);
    expect(b.shape).toEqual([3, 4]);
  });

  it("should transpose", () => {
    const a = array([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    const b = a.transpose();
    expect(b.shape).toEqual([3, 2]);
    expect(b.get(0, 0)).toBe(1);
    expect(b.get(0, 1)).toBe(4);
    expect(b.get(2, 0)).toBe(3);
  });

  it("should flatten", () => {
    const a = array([
      [1, 2],
      [3, 4],
    ]);
    const b = a.flatten();
    expect(b.shape).toEqual([4]);
    expect(Array.from(b.data)).toEqual([1, 2, 3, 4]);
  });

  it("should copy", () => {
    const a = array([1, 2, 3]);
    const b = a.copy();
    b.data[0] = 99;
    expect(a.data[0]).toBe(1);
  });

  it("should squeeze", () => {
    const a = array([[[1, 2, 3]]]);
    expect(a.shape).toEqual([1, 1, 3]);
    const b = a.squeeze();
    expect(b.shape).toEqual([3]);
  });

  it("should expand_dims", () => {
    const a = array([1, 2, 3]);
    const b = a.expandDims(0);
    expect(b.shape).toEqual([1, 3]);
    const c = a.expandDims(1);
    expect(c.shape).toEqual([3, 1]);
  });

  it("should support negative indexing", () => {
    const a = array([1, 2, 3, 4, 5]);
    expect(a.get(-1)).toBe(5);
    expect(a.get(-2)).toBe(4);
  });
});

describe("Math Operations", () => {
  it("should add arrays", () => {
    const a = array([1, 2, 3]);
    const b = array([4, 5, 6]);
    const c = np.add(a, b);
    expect(Array.from(c.data)).toEqual([5, 7, 9]);
  });

  it("should subtract arrays", () => {
    const a = array([5, 7, 9]);
    const b = array([1, 2, 3]);
    const c = np.subtract(a, b);
    expect(Array.from(c.data)).toEqual([4, 5, 6]);
  });

  it("should multiply arrays", () => {
    const a = array([1, 2, 3]);
    const b = array([4, 5, 6]);
    const c = np.multiply(a, b);
    expect(Array.from(c.data)).toEqual([4, 10, 18]);
  });

  it("should divide arrays", () => {
    const a = array([10, 20, 30]);
    const b = array([2, 4, 5]);
    const c = np.divide(a, b);
    expect(Array.from(c.data)).toEqual([5, 5, 6]);
  });

  it("should compute power", () => {
    const a = array([2, 3, 4]);
    const b = array([2, 2, 2]);
    const c = np.power(a, b);
    expect(Array.from(c.data)).toEqual([4, 9, 16]);
  });

  it("should compute sqrt", () => {
    const a = array([4, 9, 16]);
    const c = np.sqrt(a);
    expect(Array.from(c.data)).toEqual([2, 3, 4]);
  });

  it("should compute trig functions", () => {
    const a = array([0, Math.PI / 2, Math.PI]);
    const s = np.sin(a);
    expect(s.data[0]).toBeCloseTo(0);
    expect(s.data[1]).toBeCloseTo(1);
    expect(s.data[2]).toBeCloseTo(0);
  });

  it("should compute exp and log", () => {
    const a = array([0, 1, 2]);
    const e = np.exp(a);
    expect(e.data[0]).toBeCloseTo(1);
    expect(e.data[1]).toBeCloseTo(Math.E);

    const l = np.log(e);
    expect(l.data[0]).toBeCloseTo(0);
    expect(l.data[1]).toBeCloseTo(1);
    expect(l.data[2]).toBeCloseTo(2);
  });

  it("should clip values", () => {
    const a = array([1, 5, 10, 15, 20]);
    const c = np.clip(a, 5, 15);
    expect(Array.from(c.data)).toEqual([5, 5, 10, 15, 15]);
  });

  it("should compute abs", () => {
    const a = array([-1, -2, 3, -4]);
    const c = np.abs(a);
    expect(Array.from(c.data)).toEqual([1, 2, 3, 4]);
  });
});

describe("Broadcasting", () => {
  it("should broadcast scalar with array", () => {
    const a = array([1, 2, 3]);
    const b = a.add(10);
    expect(Array.from(b.data)).toEqual([11, 12, 13]);
  });

  it("should broadcast arrays of different shapes", () => {
    const a = array([
      [1, 2, 3],
      [4, 5, 6],
    ]); // (2,3)
    const b = array([10, 20, 30]); // (3,)
    const c = np.add(a, b);
    expect(c.shape).toEqual([2, 3]);
    expect(c.get(0, 0)).toBe(11);
    expect(c.get(1, 2)).toBe(36);
  });
});

describe("Statistics", () => {
  it("should compute sum", () => {
    const a = array([1, 2, 3, 4, 5]);
    expect(np.sum(a)).toBe(15);
  });

  it("should compute mean", () => {
    const a = array([1, 2, 3, 4, 5]);
    expect(np.mean(a)).toBeCloseTo(3);
  });

  it("should compute std", () => {
    const a = array([2, 4, 4, 4, 5, 5, 7, 9]);
    const s = np.std(a) as number;
    expect(s).toBeCloseTo(2, 0);
  });

  it("should compute min and max", () => {
    const a = array([3, 1, 4, 1, 5, 9]);
    expect(np.min(a)).toBe(1);
    expect(np.max(a)).toBe(9);
  });

  it("should compute median", () => {
    const a = array([1, 3, 5, 7, 9]);
    expect(np.median(a)).toBe(5);

    const b = array([1, 2, 3, 4]);
    expect(np.median(b)).toBe(2.5);
  });

  it("should compute sum along axis", () => {
    const a = array([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    const s0 = np.sum(a, 0) as NdArray;
    expect(Array.from(s0.data)).toEqual([5, 7, 9]);
    const s1 = np.sum(a, 1) as NdArray;
    expect(Array.from(s1.data)).toEqual([6, 15]);
  });

  it("should compute prod", () => {
    const a = array([1, 2, 3, 4]);
    expect(np.prod(a)).toBe(24);
  });
});

describe("Linear Algebra", () => {
  it("should compute dot product of vectors", () => {
    const a = array([1, 2, 3]);
    const b = array([4, 5, 6]);
    const c = np.dot(a, b);
    expect(c.item()).toBe(32);
  });

  it("should compute matrix multiplication", () => {
    const a = array([
      [1, 2],
      [3, 4],
    ]);
    const b = array([
      [5, 6],
      [7, 8],
    ]);
    const c = np.matmul(a, b);
    expect(c.get(0, 0)).toBe(19);
    expect(c.get(0, 1)).toBe(22);
    expect(c.get(1, 0)).toBe(43);
    expect(c.get(1, 1)).toBe(50);
  });

  it("should compute determinant", () => {
    const a = array([
      [1, 2],
      [3, 4],
    ]);
    const d = np.linalg.det(a);
    expect(d).toBeCloseTo(-2);
  });

  it("should compute inverse", () => {
    const a = array([
      [1, 2],
      [3, 4],
    ]);
    const ainv = np.linalg.inv(a);
    expect(ainv.get(0, 0)).toBeCloseTo(-2);
    expect(ainv.get(0, 1)).toBeCloseTo(1);
    expect(ainv.get(1, 0)).toBeCloseTo(1.5);
    expect(ainv.get(1, 1)).toBeCloseTo(-0.5);
  });

  it("should solve linear system", () => {
    const A = array([
      [2, 1],
      [5, 7],
    ]);
    const b = array([11, 13]);
    const x = np.linalg.solve(A, b);
    expect(x.data[0]).toBeCloseTo(7.111, 2);
    expect(x.data[1]).toBeCloseTo(-3.222, 2);
  });

  it("should compute norm", () => {
    const a = array([3, 4]);
    expect(np.linalg.norm(a)).toBeCloseTo(5);
  });

  it("should compute trace", () => {
    const a = array([
      [1, 2],
      [3, 4],
    ]);
    expect(np.linalg.trace(a)).toBe(5);
  });
});

describe("Random", () => {
  it("should generate reproducible results with seed", () => {
    np.random.seed(42);
    const a = np.random.rand(5);
    np.random.seed(42);
    const b = np.random.rand(5);
    for (let i = 0; i < 5; i++) {
      expect(a.data[i]).toBe(b.data[i]);
    }
  });

  it("should generate random integers", () => {
    const a = np.random.randint(0, 10, 100);
    for (let i = 0; i < a.size; i++) {
      expect(a.data[i]).toBeGreaterThanOrEqual(0);
      expect(a.data[i]).toBeLessThan(10);
    }
  });

  it("should generate normal distribution", () => {
    np.random.seed(0);
    const a = np.random.normal(0, 1, 10000);
    const m = np.mean(a) as number;
    const s = np.std(a) as number;
    expect(m).toBeCloseTo(0, 0);
    expect(s).toBeCloseTo(1, 0);
  });

  it("should generate uniform distribution", () => {
    const a = np.random.uniform(2, 5, 1000);
    for (let i = 0; i < a.size; i++) {
      expect(a.data[i]).toBeGreaterThanOrEqual(2);
      expect(a.data[i]).toBeLessThan(5);
    }
  });
});

describe("Array Manipulation", () => {
  it("should concatenate arrays", () => {
    const a = array([1, 2, 3]);
    const b = array([4, 5, 6]);
    const c = np.concatenate([a, b]);
    expect(Array.from(c.data)).toEqual([1, 2, 3, 4, 5, 6]);
  });

  it("should stack arrays", () => {
    const a = array([1, 2, 3]);
    const b = array([4, 5, 6]);
    const c = np.stack([a, b]);
    expect(c.shape).toEqual([2, 3]);
    expect(c.get(0, 0)).toBe(1);
    expect(c.get(1, 0)).toBe(4);
  });

  it("should vstack", () => {
    const a = array([1, 2, 3]);
    const b = array([4, 5, 6]);
    const c = np.vstack([a, b]);
    expect(c.shape).toEqual([2, 3]);
  });

  it("should hstack", () => {
    const a = array([1, 2, 3]);
    const b = array([4, 5, 6]);
    const c = np.hstack([a, b]);
    expect(c.shape).toEqual([6]);
    expect(Array.from(c.data)).toEqual([1, 2, 3, 4, 5, 6]);
  });

  it("should flip array", () => {
    const a = array([1, 2, 3, 4, 5]);
    const b = np.flip(a, 0);
    expect(Array.from(b.data)).toEqual([5, 4, 3, 2, 1]);
  });

  it("should find unique elements", () => {
    const a = array([3, 1, 2, 1, 3, 2, 4]);
    const u = np.unique(a);
    expect(Array.from(u.data)).toEqual([1, 2, 3, 4]);
  });

  it("should compute where", () => {
    const condition = array([1, 0, 1, 0, 1]);
    const x = array([10, 20, 30, 40, 50]);
    const y = array([0, 0, 0, 0, 0]);
    const result = np.where(condition, x, y);
    expect(Array.from(result.data)).toEqual([10, 0, 30, 0, 50]);
  });

  it("should compute diff", () => {
    const a = array([1, 3, 6, 10]);
    const d = np.diff(a);
    expect(Array.from(d.data)).toEqual([2, 3, 4]);
  });

  it("should compute cumsum", () => {
    const a = array([1, 2, 3, 4, 5]);
    const c = np.cumsum(a);
    expect(Array.from(c.data)).toEqual([1, 3, 6, 10, 15]);
  });

  it("should compute outer product", () => {
    const a = array([1, 2, 3]);
    const b = array([4, 5]);
    const c = np.outer(a, b);
    expect(c.shape).toEqual([3, 2]);
    expect(c.get(0, 0)).toBe(4);
    expect(c.get(0, 1)).toBe(5);
    expect(c.get(2, 0)).toBe(12);
    expect(c.get(2, 1)).toBe(15);
  });

  it("should compute cross product", () => {
    const a = array([1, 0, 0]);
    const b = array([0, 1, 0]);
    const c = np.cross(a, b);
    expect(Array.from(c.data)).toEqual([0, 0, 1]);
  });
});

describe("Slicing", () => {
  it("should slice 1D array", () => {
    const a = array([0, 1, 2, 3, 4, 5]);
    const b = a.slice([1, 4]);
    expect(Array.from(b.data)).toEqual([1, 2, 3]);
  });

  it("should slice with step", () => {
    const a = array([0, 1, 2, 3, 4, 5]);
    const b = a.slice([0, 6, 2]);
    expect(Array.from(b.data)).toEqual([0, 2, 4]);
  });

  it("should slice 2D array", () => {
    const a = arange(12).reshape([3, 4]);
    const b = a.slice([0, 2], [1, 3]);
    expect(b.shape).toEqual([2, 2]);
    expect(b.get(0, 0)).toBe(1);
    expect(b.get(0, 1)).toBe(2);
    expect(b.get(1, 0)).toBe(5);
    expect(b.get(1, 1)).toBe(6);
  });
});

describe("Sorting", () => {
  it("should sort array", () => {
    const a = array([3, 1, 4, 1, 5, 9, 2, 6]);
    const sorted = a.sort();
    expect(Array.from(sorted.data)).toEqual([1, 1, 2, 3, 4, 5, 6, 9]);
  });

  it("should argsort", () => {
    const a = array([30, 10, 20]);
    const indices = a.argsort();
    expect(Array.from(indices.data)).toEqual([1, 2, 0]);
  });
});

describe("Type casting", () => {
  it("should cast dtype", () => {
    const a = array([1.5, 2.7, 3.9]);
    const b = a.astype("int32");
    expect(b.dtype).toBe("int32");
    expect(b.data[0]).toBe(1);
    expect(b.data[1]).toBe(2);
    expect(b.data[2]).toBe(3);
  });
});

describe("String representation", () => {
  it("should convert to string", () => {
    const a = array([1, 2, 3]);
    expect(a.toString()).toBe("[1, 2, 3]");
  });

  it("should convert to nested list", () => {
    const a = array([
      [1, 2],
      [3, 4],
    ]);
    expect(a.tolist()).toEqual([
      [1, 2],
      [3, 4],
    ]);
  });
});
