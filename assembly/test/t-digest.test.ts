import { Centroid } from "../centroid";

import { mergeData, tDigestCluster, estimateQuantile } from "../t-digest";
import { centroidsFromFloat64Array } from "../utils";

describe("mergeData", () => {
  it("total weight must be correct", () => {
    const N = 10000;
    const X = new Float64Array(N).map((_, i) => ((1.0 * i) / N) ** 2);
    const C = mergeData(centroidsFromFloat64Array(X));
    // const weight = ;
    expect(C.weight).toBe(N);
  });
});

function uniformPoints(N: i32, lower: f64, upper: f64): Float64Array {
  const range = upper - lower;
  const x = new Float64Array(N);
  for (let i = 0; i < N; i++) {
    x[i] = (range * (i + 1) + lower) / (N - 1);
  }
  return x;
}

export function shuffleArray(arr: Float64Array): Float64Array {
  const N = arr.length;
  for (let i = 0, j = N - 1; i < N; i += 2, j--) {
    const tmp = arr[i];
    arr[i] = arr[j];
    arr[j] = tmp;
  }
  return arr;
}

function logisticInvCdf(y: f64, mu: f64 = 0.0, s: f64 = 1.0): f64 {
  return mu - s * Math.log(1 / y - 1);
}

function logisticPoints(N: i32, mu: f64 = 0.0, s: f64 = 1.0): Float64Array {
  const x = new Float64Array(N);
  for (let i = 0; i < N; i++) {
    x[i] = logisticInvCdf((1.0 * i + 1) / (N + 1), mu, s);
  }
  return x;
}

describe("estimateQuantile from mergeData", () => {
  it("correct approximate quantiles for Uniform(0,5) distribution", () => {
    const N = 3000;
    const X = shuffleArray(uniformPoints(N, 0, 5));

    const C = mergeData(centroidsFromFloat64Array(X), null, 100);
    const x = estimateQuantile(C, 0.5);
    expect<f64>(estimateQuantile(C, 0.5)).toBeCloseTo(2.5);
    expect<f64>(estimateQuantile(C, 0.1)).toBeCloseTo(0.5);
    expect<f64>(estimateQuantile(C, 0.8)).toBeCloseTo(4.0);
  });

  it("correct approximate quantiles for logistic(0,1) distribution", () => {
    const N = 1000;
    const X = shuffleArray(logisticPoints(N, 0, 1));
    const C = mergeData(centroidsFromFloat64Array(X), null, 100);

    expect<f64>(estimateQuantile(C, 0.5)).toBeCloseTo(logisticInvCdf(0.5));
    expect<f64>(estimateQuantile(C, 0.05) / logisticInvCdf(0.05)).toBeCloseTo(
      1.0
    );
    expect<f64>(estimateQuantile(C, 0.95) / logisticInvCdf(0.95)).toBeCloseTo(
      1.0
    );
    expect<f64>(estimateQuantile(C, 0.99) / logisticInvCdf(0.99)).toBeCloseTo(
      1.0
    );
  });

  it("correct approximate quantiles for logistic(0,1) distribution (larger sample)", () => {
    const N = 5000;
    const X = shuffleArray(logisticPoints(N, 0, 1));
    // const X = logisticPoints(N, 0, 1);
    const C = mergeData(centroidsFromFloat64Array(X), null, 200);

    expect<f64>(estimateQuantile(C, 0.5)).toBeCloseTo(logisticInvCdf(0.5));
    expect<f64>(estimateQuantile(C, 0.05) / logisticInvCdf(0.05)).toBeCloseTo(
      1.0
    );
    expect<f64>(estimateQuantile(C, 0.95) / logisticInvCdf(0.95)).toBeCloseTo(
      1.0
    );
    expect<f64>(estimateQuantile(C, 0.99) / logisticInvCdf(0.99)).toBeCloseTo(
      1.0
    );
  });
});

describe("estimateQuantile from tDigestCluster", () => {
  it("correct approximate quantiles for Uniform(0,5) distribution", () => {
    const N = 8000;
    const X = shuffleArray(uniformPoints(N, 0, 5));
    const C = mergeData(centroidsFromFloat64Array(X));

    const N2 = 5036;
    const X2 = shuffleArray(uniformPoints(N2, 0, 5));
    const newPoints = centroidsFromFloat64Array(X2);

    const C3 = tDigestCluster(newPoints, C);

    expect<f64>(estimateQuantile(C3, 0.5)).toBeCloseTo(2.5);
    expect<f64>(estimateQuantile(C3, 0.1)).toBeCloseTo(0.5);
    expect<f64>(estimateQuantile(C3, 0.8)).toBeCloseTo(4.0);
  });
});

describe("estimateQuantile from naive tDigestCluster for logistic(0,1) ", () => {
  // FIXME: not having closures is a PITA, means we have to copy this all out by hand instead of using one DRY loop
  // [0.05, 0.1, 0.5, 0.95, 0.99].forEach((q) => {
  //   it(
  //     "correct approximate quantiles distribution for logistic(0,1) at q=" +
  //       q.toString(),
  //     () => {
  //       const N = 1000;
  //       const X = shuffleArray(logisticPoints(N));
  //       const C = mergeData(centroidsFromFloat64Array(X));

  //       const N2 = 15036;
  //       const X2 = shuffleArray(logisticPoints(N2));
  //       const newPoints = centroidsFromFloat64Array(X2);

  //       const C3 = tDigestCluster(newPoints, C);
  //       expect(estimateQuantile(C3, q)).toBeCloseTo(logisticInvCdf(q));
  //     }
  //   );
  // });

  it("correct approximate quantiles distribution for logistic(0,1) at q=0.01", () => {
    const N = 1000;
    const X = shuffleArray(logisticPoints(N));
    const C = mergeData(centroidsFromFloat64Array(X), null, 500);

    const N2 = 15036;
    const X2 = shuffleArray(logisticPoints(N2));
    const newPoints = centroidsFromFloat64Array(X2);

    const C3 = tDigestCluster(newPoints, C, 500);
    expect(estimateQuantile(C3, 0.01)).toBeCloseTo(logisticInvCdf(0.01));
  });

  it("correct approximate quantiles distribution for logistic(0,1) at q=0.05", () => {
    const N = 1000;
    const X = shuffleArray(logisticPoints(N));
    const C = mergeData(centroidsFromFloat64Array(X), null, 500);

    const N2 = 15036;
    const X2 = shuffleArray(logisticPoints(N2));
    const newPoints = centroidsFromFloat64Array(X2);

    const C3 = tDigestCluster(newPoints, C, 500);
    expect(estimateQuantile(C3, 0.05)).toBeCloseTo(logisticInvCdf(0.05));
  });
  it("correct approximate quantiles distribution for logistic(0,1) at q=0.15", () => {
    const N = 1000;
    const X = shuffleArray(logisticPoints(N));
    const C = mergeData(centroidsFromFloat64Array(X), null, 500);

    const N2 = 15036;
    const X2 = shuffleArray(logisticPoints(N2));
    const newPoints = centroidsFromFloat64Array(X2);

    const C3 = tDigestCluster(newPoints, C, 500);
    expect(estimateQuantile(C3, 0.15)).toBeCloseTo(logisticInvCdf(0.15));
  });
  it("correct approximate quantiles distribution for logistic(0,1) at q=0.5", () => {
    const N = 1000;
    const X = shuffleArray(logisticPoints(N));
    const C = mergeData(centroidsFromFloat64Array(X), null, 500);

    const N2 = 15036;
    const X2 = shuffleArray(logisticPoints(N2));
    const newPoints = centroidsFromFloat64Array(X2);

    const C3 = tDigestCluster(newPoints, C, 500);
    expect(estimateQuantile(C3, 0.5)).toBeCloseTo(logisticInvCdf(0.5));
  });
  it("correct approximate quantiles distribution for logistic(0,1) at q=0.75", () => {
    const N = 1000;
    const X = shuffleArray(logisticPoints(N));
    const C = mergeData(centroidsFromFloat64Array(X), null, 500);

    const N2 = 15036;
    const X2 = shuffleArray(logisticPoints(N2));
    const newPoints = centroidsFromFloat64Array(X2);

    const C3 = tDigestCluster(newPoints, C, 500);
    expect(estimateQuantile(C3, 0.75)).toBeCloseTo(logisticInvCdf(0.75));
  });

  it("correct approximate quantiles distribution for logistic(0,1) at q=0.9", () => {
    const N = 1000;
    const X = shuffleArray(logisticPoints(N));
    const C = mergeData(centroidsFromFloat64Array(X), null, 500);

    const N2 = 15036;
    const X2 = shuffleArray(logisticPoints(N2));
    const newPoints = centroidsFromFloat64Array(X2);

    const C3 = tDigestCluster(newPoints, C, 500);
    expect(estimateQuantile(C3, 0.9)).toBeCloseTo(logisticInvCdf(0.9));
  });
});
