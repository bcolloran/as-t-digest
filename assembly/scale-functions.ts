export type scaleFunction = (q: f64, delta: f64) => f64;

export const k0ScaleFunction = (q: f64, delta: f64): f64 => {
  return q + 2.0 / delta;
};

export const kFn_1 = (q: f64, delta: f64): f64 =>
  (delta / (2.0 * Math.PI)) * Math.asin(2.0 * q - 1.0);

export const kInvFn_1 = (k: f64, delta: f64): f64 =>
  0.5 * (1 + Math.sin((2 * Math.PI * k) / delta));

export const k1ScaleFunction = (q: f64, delta: f64): f64 => {
  return kInvFn_1(kFn_1(q, delta) + 1.0, delta);
};

export const KInvOfKFn_1 = (q: f64, delta: f64): f64 =>
  0.5 + 0.5 * Math.sin(Math.asin(2 * q - 1) + (2 * Math.PI) / delta);
