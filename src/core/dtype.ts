/**
 * Dtype system — typed array constructors and metadata
 */

export type DtypeString =
  | "float32"
  | "float64"
  | "int8"
  | "int16"
  | "int32"
  | "uint8"
  | "uint16"
  | "uint32"
  | "bool";

export type TypedArrayConstructor =
  | Float32ArrayConstructor
  | Float64ArrayConstructor
  | Int8ArrayConstructor
  | Int16ArrayConstructor
  | Int32ArrayConstructor
  | Uint8ArrayConstructor
  | Uint16ArrayConstructor
  | Uint32ArrayConstructor;

export type TypedArray =
  | Float32Array
  | Float64Array
  | Int8Array
  | Int16Array
  | Int32Array
  | Uint8Array
  | Uint16Array
  | Uint32Array;

export interface DtypeInfo {
  name: DtypeString;
  bytes: number;
  ArrayType: TypedArrayConstructor;
}

export const dtypes: Record<DtypeString, DtypeInfo> = {
  float32: { name: "float32", bytes: 4, ArrayType: Float32Array },
  float64: { name: "float64", bytes: 8, ArrayType: Float64Array },
  int8: { name: "int8", bytes: 1, ArrayType: Int8Array },
  int16: { name: "int16", bytes: 2, ArrayType: Int16Array },
  int32: { name: "int32", bytes: 4, ArrayType: Int32Array },
  uint8: { name: "uint8", bytes: 1, ArrayType: Uint8Array },
  uint16: { name: "uint16", bytes: 2, ArrayType: Uint16Array },
  uint32: { name: "uint32", bytes: 4, ArrayType: Uint32Array },
  bool: { name: "bool", bytes: 1, ArrayType: Uint8Array },
};

export const getDtypeInfo = (dtype: DtypeString): DtypeInfo => {
  const info = dtypes[dtype];
  if (!info) throw new Error(`Unknown dtype: ${dtype}`);
  return info;
};

export const inferDtype = (value: number | boolean): DtypeString =>
  typeof value === "boolean"
    ? "bool"
    : Number.isInteger(value)
      ? "int32"
      : "float64";

export const createTypedArray = (
  dtype: DtypeString,
  size: number,
): TypedArray => new (getDtypeInfo(dtype).ArrayType)(size);

export const castTypedArray = (
  data: TypedArray,
  dtype: DtypeString,
): TypedArray => {
  const { ArrayType } = getDtypeInfo(dtype);
  if (data instanceof ArrayType) return data;
  const out = new ArrayType(data.length);
  for (let i = 0; i < data.length; i++) out[i] = data[i];
  return out;
};
