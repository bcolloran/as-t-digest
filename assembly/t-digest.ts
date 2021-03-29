import { interpolate, findMergeCentroid } from "./utils";
import { scaleFunction, KInvOfKFn_1, kFn_1 } from "./scale-functions";
import { AvlTreeMap, TreeIterator } from "as-avl-tree";
import { Centroid } from "./centroid";

class TDigest {
  private _tree: AvlTreeMap<f64, Centroid>;
  private _weight: f64;
  private _iter: TreeIterator<f64> | null = null;

  constructor(public size: f64 = 0, public centroids: Centroid[] = []) {
    this._tree = new AvlTreeMap<f64, Centroid>();
  }

  insert(c: Centroid): void {
    this._tree.insert(c.mean, c);
    this._weight += c.count;
  }

  initIterator(): void {
    this._iter = new TreeIterator(this._tree.tree);
  }

  next(): Centroid {
    const iter = this._iter;
    if (iter == null) {
      throw new Error("must `.initIterator()` before calling `.next()");
    }
    const key = iter.next();
    return this._tree.get(key);
  }
  hasNext(): bool {
    const iter = this._iter;
    if (iter == null) return false;
    return iter.hasNext();
  }

  get weight(): f64 {
    return this._weight;
  }

  getMinCentroid(): Centroid {
    return this._tree.findLeftmostValue();
  }
  getMaxCentroid(): Centroid {
    return this._tree.findRightmostValue();
  }

  getClosestCentroid(target: f64): Centroid {
    return this._tree.getClosestValue(target);
  }
}

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
  tDigest: TDigest | null = null,
  delta: f64 = 100.0,
  KInvOfKFn: scaleFunction = KInvOfKFn_1
): TDigest {
  if (tDigest == null) tDigest = new TDigest();

  for (let i = 0; i < newPoints.length; i++) {
    tDigest.insert(newPoints[i]);
  }
  const S = tDigest.weight;
  let Cprime = new TDigest();
  let q_0 = 0.0;
  let q_lim = KInvOfKFn(q_0, delta);

  tDigest.initIterator();

  let sigma = tDigest.next();

  while (tDigest.hasNext()) {
    const x = tDigest.next();
    let q = q_0 + (sigma.count + x.count) / S;
    if (q <= q_lim) {
      sigma = sigma + x;
    } else {
      Cprime.insert(sigma);
      q_0 += sigma.count / S;
      q_lim = KInvOfKFn(q_0, delta);
      sigma = x;
    }
  }
  Cprime.insert(sigma);
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
  tDigest: TDigest | null,
  delta: f64 = 100.0,
  growthBound: f64 = 5.0,
  KInvOfKFn: scaleFunction = KInvOfKFn_1
  // kFn: scaleFunction = kFn_1
): TDigest {
  tDigest = tDigest == null ? new TDigest() : tDigest;
  let totalWeight = tDigest.weight;

  let addCount = 0;
  let insertCount = 0;

  for (let i = 0, N = newPoints.length; i < N; i++) {
    const x = newPoints[i];
    let closestCentroid = tDigest.getClosestCentroid(x.mean);

    // FIXME not using t-digest weighting scheme, letting the merge step handle that
    if ((closestCentroid.count + x.count) / totalWeight < 1.0 / delta) {
      closestCentroid.updateAdd(x);
      addCount++;
    } else {
      tDigest.insert(x);
      insertCount++;
    }

    if (totalWeight > growthBound * delta) {
      mergeData([], tDigest, delta, KInvOfKFn);
    }
  }
  return mergeData([], tDigest, delta, KInvOfKFn);
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
export function estimateQuantile(tDigest: TDigest, q: f64): f64 {
  if (q == 0) return tDigest.getMinCentroid().mean;
  if (q == 1) return tDigest.getMaxCentroid().mean;

  const S = tDigest.weight;
  const targetWeight = S * q;

  tDigest.initIterator();

  let leftCentroid = tDigest.next();
  let rightCentroid = tDigest.next();
  // for the first interval (between the first 2 points), there is 0 weight to the left of the interval's left end
  let weightLessThanLeft = 0.0;
  // for the first interval (between the first 2 points), to the left of the interval's right end is the full weight of the left centroid and half the weight of the right
  let weightLessThanRight = leftCentroid.count + 0.5 * rightCentroid.count;

  while (tDigest.hasNext()) {
    if (targetWeight < weightLessThanRight) {
      return interpolate(
        targetWeight,
        weightLessThanLeft,
        weightLessThanRight,
        leftCentroid.mean,
        rightCentroid.mean
      );
    } else {
      // step to the next interval;
      leftCentroid = rightCentroid;
      rightCentroid = tDigest.next();

      weightLessThanLeft = weightLessThanRight;
      // this new interval contains half of the weight of both of its bounding centroids, so add those to the running sum for the right end of the interval
      weightLessThanRight += 0.5 * (leftCentroid.count + rightCentroid.count);
    }
  }

  // q is in the last interval between N-2 and N-1
  return interpolate(
    targetWeight,
    weightLessThanLeft,
    S,
    leftCentroid.mean,
    rightCentroid.mean
  );
}
