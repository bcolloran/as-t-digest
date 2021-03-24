import { Centroid } from "./t-digest-naive";
import { scaleFunction } from "./scale-functions";

export function interpolate(y: f64, y1: f64, y2: f64, x1: f64, x2: f64): f64 {
  return x1 + ((x2 - x1) / (y2 - y1)) * (y - y1);
}

export function centroidsFromPoints(points: f64[]): Centroid[] {
  return points.map<Centroid>((x) => new Centroid(x, 1));
}

export function centroidsFromFloat64Array(points: Float64Array): Centroid[] {
  const centroids: Centroid[] = [];
  for (let i = 0, N = points.length; i < N; i++) {
    centroids.push(new Centroid(points[i], 1));
  }
  // points.map((x) => centroids.push(new Centroid(x, 1)));
  return centroids;
}

export function findClosestCentroids(
  x: Centroid,
  centroids: Centroid[]
): Centroid[] {
  if (centroids.length == 0) return [];

  let minDist = Infinity;
  let bestIndex = 0;

  for (let i = 0, N = centroids.length; i < N; i++) {
    let dist = abs(centroids[i].mean - x.mean);
    if (dist <= minDist) {
      bestIndex = i;
      minDist = dist;
    } else {
      break;
    }
  }

  // in case of a tie
  if (
    bestIndex - 1 >= 0 &&
    abs(centroids[bestIndex - 1].mean - x.mean) == minDist
  ) {
    return [centroids[bestIndex - 1], centroids[bestIndex]];
  }
  return [centroids[bestIndex]];
}

export function findMergeCentroid(
  x: Centroid,
  centroids: Centroid[],
  totalWeight: f64,
  kFn: scaleFunction,
  delta: f64
): Centroid | null {
  if (centroids.length == 0) return null;

  let minDist = Infinity;
  let bestIndex = 0;
  let W_left = 0.0;

  for (let i = 0, N = centroids.length; i < N; i++) {
    let dist = abs(centroids[i].mean - x.mean);
    if (dist <= minDist) {
      bestIndex = i;
      minDist = dist;
      if (i > 0) W_left += centroids[i].count;
    } else {
      break;
    }
  }

  let mergeCentroid: Centroid | null = null;

  let q_left = W_left / totalWeight;
  let q_right = q_left + (centroids[bestIndex].count + x.count) / totalWeight;

  // if weight bound is satisfied, set this as the best centroid
  if (kFn(q_right, delta) - kFn(q_left, delta)) {
    mergeCentroid = centroids[bestIndex];
  }

  // in case of a distance tie...
  if (
    bestIndex - 1 >= 0 &&
    abs(centroids[bestIndex - 1].mean - x.mean) == minDist
  ) {
    // ...if the previous centroid also satisfies the weight bound and has higher weight, return it
    q_left = (W_left - centroids[bestIndex].count) / totalWeight;
    q_right = q_left + (centroids[bestIndex - 1].count + x.count) / totalWeight;
    if (
      kFn(q_right, delta) - kFn(q_left, delta) &&
      centroids[bestIndex - 1].count > centroids[bestIndex].count
    ) {
      mergeCentroid = centroids[bestIndex];
    }
  }

  return mergeCentroid;
}
