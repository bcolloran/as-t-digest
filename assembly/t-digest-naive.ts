import { interpolate, findMergeCentroid } from "./utils";
import { scaleFunction, KInvOfKFn_1, kFn_1 } from "./scale-functions";
import { Centroid, centroidSortFn } from "./centroid";

/**
 * Algorithm 1, https://arxiv.org/pdf/1902.04023.pdf p.9
 *
 * @param newPoints
 * @param centroids
 * @param delta
 * @param KInvOfKFn
 * @returns
 */
export function mergeData(
  newPoints: Centroid[],
  centroids: Centroid[] = [],
  delta: f64 = 100.0,
  KInvOfKFn: scaleFunction = KInvOfKFn_1
): Centroid[] {
  let X = centroids.concat(newPoints);
  X = X.sort(centroidSortFn);
  const S = X.reduce((s, centroid) => s + centroid.count, 0.0);
  let Cprime: Centroid[] = [];
  let q_0 = 0.0;
  let q_lim = KInvOfKFn(q_0, delta);
  let sigma = X[0];
  for (let i = 1, N = X.length; i < N; i++) {
    let q = q_0 + (sigma.count + X[i].count) / S;
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

/**
 * Algorithm 2, https://arxiv.org/pdf/1902.04023.pdf p.12
 *
 * @param newPoints
 * @param centroids
 * @param delta
 * @param growthBound
 * @param KInvOfKFn
 * @returns
 */
export function tDigestCluster(
  newPoints: Centroid[],
  centroids: Centroid[] = [],
  delta: f64 = 100.0,
  growthBound: f64 = 5.0,
  KInvOfKFn: scaleFunction = KInvOfKFn_1,
  kFn: scaleFunction = kFn_1
): Centroid[] {
  let totalWeight = centroids.reduce((s, c) => s + c.count, 0.0);

  for (let i = 0, N = newPoints.length; i < N; i++) {
    const x = newPoints[i];
    let mergePoint = findMergeCentroid(x, centroids, totalWeight, kFn, delta);
    if (mergePoint) {
      mergePoint.updateAdd(x);
    } else {
      centroids.push(x);
    }
    totalWeight += x.count;

    if (totalWeight > growthBound * delta) {
      centroids = mergeData(centroids, [], delta, KInvOfKFn);
    }
  }
  return mergeData(centroids, [], delta, KInvOfKFn);
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
 * @param X A sorted array of centroids (i.e., a t-digest)
 * @param q - The target quantile; a float in the interval [0,1]
 */
export function estimateQuantile(X: Centroid[], q: f64): f64 {
  if (q == 0) return X[0].mean;

  const N = X.length;
  if (q == 1) return X[N - 1].mean;

  const S = X.reduce((s, centroid) => s + centroid.count, 0.0);
  const targetWeight = S * q;
  let weightIntervalLeft = 0.0;
  let weightIntervalRight = X[0].count + 0.5 * X[1].count;

  for (let i = 1; i < N - 1; i++) {
    if (targetWeight < weightIntervalRight) {
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
