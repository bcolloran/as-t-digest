import {
  Centroid,
  centroidSortFn,
  mergeData,
  tDigestCluster,
  estimateQuantile,
} from "../t-digest-naive";
import { centroidsFromFloat64Array } from "../utils";

describe("centroids", () => {
  it("instantiate correctly", () => {
    const c1 = new Centroid(1, 2);
    expect<f64>(c1.mean).toBe(1);
    expect<f64>(c1.count).toBe(2);
  });

  it("add correctly, case 1", () => {
    const c1 = new Centroid(1, 2) + new Centroid(1, 2);
    expect<f64>(c1.mean).toBe(1);
    expect<f64>(c1.count).toBe(4);
  });

  it("add correctly, case 2", () => {
    const c1 = new Centroid(-1, 2) + new Centroid(1, 2);
    expect<f64>(c1.mean).toBe(0);
    expect<f64>(c1.count).toBe(4);
  });

  it("add correctly, case 3", () => {
    const c1 = new Centroid(0, 1) + new Centroid(4, 3);
    expect<f64>(c1.mean).toBe(3);
    expect<f64>(c1.count).toBe(4);
  });
});

describe("mergeData", () => {
  it("total weight must be correct", () => {
    const N = 10000;
    const X = new Float64Array(N).map((_, i) => ((1.0 * i) / N) ** 2);
    const C = mergeData(centroidsFromFloat64Array(X));
    const weight = C.reduce((x, c) => x + c.count, 0.0);
    expect(weight).toBe(N);
  });
});

describe("tDigestCluster", () => {
  it("total weight must be correct", () => {
    const N = 10000;

    const X = new Float64Array(N).map((_, i) => ((1.0 * i) / N) ** 2);
    const C = mergeData(centroidsFromFloat64Array(X));

    const N2 = 5036;
    const X2 = new Float64Array(N2)
      .fill(0)
      .map((_, i) => ((1.0 * i) / N2) ** 2);
    const newPoints = centroidsFromFloat64Array(X2);

    const C3 = tDigestCluster(newPoints, C);
    const weight = C3.reduce((x, c) => x + c.count, 0.0);
    expect(weight).toBe(N + N2);
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

// function shuffleArray(arr: Float64Array): Float64Array {
//   const N = arr.length;
//   let tmp: f64;
//   for (let i = 0, j = N - 1; i < N / 2; i++, j--) {
//     tmp = arr[i * 2];
//     arr[i * 2] = arr[j];
//     arr[j] = tmp;
//   }
//   return arr;
// }

export function shuffleArray(arr: Float64Array): Float64Array {
  const N = arr.length;
  for (let i = 0, j = N - 1; i < N; i += 2, j--) {
    const tmp = arr[i];
    arr[i] = arr[j];
    arr[j] = tmp;
  }
  return arr;
}

// export function shuffleArray(arr: Float64Array): Float64Array {
//   const N = arr.length;
//   const arrOut = new Float64Array(N);
//   for (let i = 0; i < N; i++) {
//     arrOut[i] = arr[i];
//   }
//   for (let i = 0, j = N - 1; i < N; i += 2, j--) {
//     arrOut[i] = arr[j];
//     arrOut[j] = arr[i];
//   }
//   return arrOut;
// }

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

    const C = mergeData(centroidsFromFloat64Array(X), [], 100);
    const x = estimateQuantile(C, 0.5);
    expect<f64>(estimateQuantile(C, 0.5)).toBeCloseTo(2.5);
    expect<f64>(estimateQuantile(C, 0.1)).toBeCloseTo(0.5);
    expect<f64>(estimateQuantile(C, 0.8)).toBeCloseTo(4.0);
  });

  it("correct approximate quantiles for logistic(0,1) distribution", () => {
    const N = 1000;
    const X = shuffleArray(logisticPoints(N, 0, 1));
    const C = mergeData(centroidsFromFloat64Array(X), [], 100);

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
    const C = mergeData(centroidsFromFloat64Array(X), [], 200);

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
    const C = mergeData(centroidsFromFloat64Array(X), [], 100);

    const N2 = 5036;
    const X2 = shuffleArray(uniformPoints(N2, 0, 5));
    const newPoints = centroidsFromFloat64Array(X2);

    const C3 = tDigestCluster(newPoints, C);

    // const x = estimateQuantile(C, 0.5);
    expect<f64>(estimateQuantile(C3, 0.5)).toBeCloseTo(2.5);
    expect<f64>(estimateQuantile(C3, 0.1)).toBeCloseTo(0.5);
    expect<f64>(estimateQuantile(C3, 0.8)).toBeCloseTo(4.0);
  });

  it("correct approximate quantiles for logistic(0,1) distribution", () => {
    const N = 1000;
    // const X = new Float64Array(N)
    //   .fill(0)
    //   .map((_, i) => (1.0 * i + 1) / (N + 1))
    //   .map((x) => logisticInvCdf(x, 0, 1));
    const X = shuffleArray(logisticPoints(N, 0, 1));

    const C = mergeData(centroidsFromFloat64Array(X), [], 100);

    const N2 = 5036;
    const X2 = shuffleArray(logisticPoints(N2, 0, 1));
    const newPoints = centroidsFromFloat64Array(X2);

    const C3 = tDigestCluster(newPoints, C);

    expect(estimateQuantile(C3, 0.5)).toBeCloseTo(logisticInvCdf(0.5));
    expect(estimateQuantile(C3, 0.05) / logisticInvCdf(0.05)).toBeCloseTo(1.0);
    expect(estimateQuantile(C3, 0.95) / logisticInvCdf(0.95)).toBeCloseTo(1.0);
    expect(estimateQuantile(C3, 0.99) / logisticInvCdf(0.99)).toBeCloseTo(1.0);
  });
});
