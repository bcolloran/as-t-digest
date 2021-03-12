export class Centroid {
  constructor(public mean: f64, public count: f64) {}

  @operator("+")
  static add(a: Centroid, b: Centroid): Centroid {
    const count = a.count + b.count;
    const mean = (a.count * a.mean + b.count * b.mean) / count;
    return new Centroid(mean, count);
  }
}

export function centroidsFromPoints(points: f64[]): Centroid[] {
  return points.map<Centroid>((x) => new Centroid(x, 1));
}

export function centroidSortFn(c1: Centroid, c2: Centroid): i32 {
  return c1.mean < c2.mean ? -1 : c1.mean === c2.mean ? 0 : 1;
}

export type scaleFunction = (q: f64, delta: f64) => f64;

function k0ScaleFunction(q: f64, delta: f64): f64 {
  return q + 2.0 / delta;
}

const k1Fn = (q: f64, delta: f64): f64 =>
  (delta / (2.0 * Math.PI)) * Math.asin(2.0 * q - 1.0);

const k1InvFn = (k: f64, delta: f64): f64 =>
  0.5 * (1 + Math.sin((2 * Math.PI * k) / delta));

function k1ScaleFunction(q: f64, delta: f64): f64 {
  return k1InvFn(k1Fn(q, delta) + 1.0, delta);
}

const KInvOfKFn_1 = (q: f64, delta: f64): f64 =>
  0.5 + 0.5 * Math.sin(Math.asin(2 * q - 1) + (2 * Math.PI) / delta);

export function mergeData(
  newPoints: Centroid[],
  centroids: Centroid[] = [],
  delta: f64 = 100.0,
  KInvOfKFn: scaleFunction = KInvOfKFn_1
): Centroid[] {
  let X = centroids.concat(newPoints);
  X = X.sort(centroidSortFn);
  const S = X.reduce((s, centroid) => s + centroid.count, 0.0);
  // log<f64>(S);
  let Cprime: Centroid[] = [];
  let q_0 = 0.0;
  let q_lim = KInvOfKFn(q_0, delta);
  // log<f64>(q);
  // log<f64>(q_lim);
  let sigma = X[0];
  for (let i = 1, N = X.length; i < N; i++) {
    let q = q_0 + (sigma.count + X[i].count) / S;
    // log<f64>(q);
    // log<f64>(q_lim);
    if (q <= q_lim) {
      sigma = sigma + X[i];
    } else {
      Cprime.push(sigma);
      q_0 += sigma.count / S;
      q_lim = KInvOfKFn(q_0, delta);
      sigma = X[i];
    }
  }
  Cprime.push(sigma);
  return Cprime;
}

export function interpolate(y: f64, y1: f64, y2: f64, x1: f64, x2: f64): f64 {
  return x1 + ((x2 - x1) / (y2 - y1)) * (y - y1);
}

/**
 * Returns a target quantile given a sorted t-digest. Note that we omit the
 * additional handling of unit weight intervals, instead using the simplifying
 * assumption that interval between *any* two consecutive centroid has weight
 * equal to half the total weight of those centroids (in otherwords, each
 * centroid has half of it's weight on each side of its mean). The only
 * exceptions are the first and last "centroids", which contribute all of their
 * weight to the following/preceding interval.
 *
 * @param X - A sorted array of centroids (i.e., a t-digest)
 * @param q - The target quantile; a float in the interval [0,1]
 */
export function estimateQuantile(X: Centroid[], q: f64): f64 {
  if (q == 0) return X[0].mean;

  const N = X.length;
  if (q == 1) return X[N - 1].mean;

  const S = X.reduce((s, centroid) => s + centroid.count, 0.0);
  // log("S inner:" + S.toString());

  const targetWeight = S * q;
  let weightIntervalLeft = 0.0;
  let weightIntervalRight = X[0].count + 0.5 * X[1].count;

  for (let i = 1; i < N - 1; i++) {
    // weightIntervalRight += ;
    if (targetWeight < weightIntervalRight) {
      // log("targetWeight,   " + targetWeight.toString());
      // log("weightIntervalLeft,   " + weightIntervalLeft.toString());
      // log("weightIntervalRight,   " + weightIntervalRight.toString());
      // log("X[i - 1].mean,   " + X[i - 1].mean.toString());
      // log("X[i].mean   " + X[i].mean.toString());
      return interpolate(
        targetWeight,
        weightIntervalLeft,
        weightIntervalRight,
        X[i - 1].mean,
        X[i].mean
      );
    } else {
      weightIntervalLeft = weightIntervalRight;
      weightIntervalRight += 0.5 * (X[i].count + X[i + 1].count);
    }
  }

  // q is in the last interval between N-2 and N-1
  return interpolate(
    targetWeight,
    weightIntervalLeft,
    S,
    X[N - 2].mean,
    X[N - 1].mean
  );
}
