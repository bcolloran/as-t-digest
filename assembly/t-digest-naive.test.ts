import {
  Centroid,
  centroidsFromPoints,
  centroidSortFn,
  mergeData,
} from "./t-digest-naive";

describe("centroids", () => {
  it("instantiate correctly", () => {
    const c1 = new Centroid(1, 2);
    expect<f64>(c1.mean).toBe(1);
    expect<f64>(c1.count).toBe(2);
  });

  it("add correctly 1", () => {
    const c1 = new Centroid(1, 2) + new Centroid(1, 2);
    expect<f64>(c1.mean).toBe(1);
    expect<f64>(c1.count).toBe(4);
  });

  it("add correctly 2", () => {
    const c1 = new Centroid(-1, 2) + new Centroid(1, 2);
    expect<f64>(c1.mean).toBe(0);
    expect<f64>(c1.count).toBe(4);
  });

  it("add correctly 3", () => {
    const c1 = new Centroid(0, 1) + new Centroid(4, 3);
    expect<f64>(c1.mean).toBe(3);
    expect<f64>(c1.count).toBe(4);
  });
});

describe("centroidsFromPoints", () => {
  it("centroidsFromPoints works correctly", () => {
    const c1 = centroidsFromPoints([1, 2, 3, 4, 5]);
    const c2 = [
      new Centroid(1, 1),
      new Centroid(2, 1),
      new Centroid(3, 1),
      new Centroid(4, 1),
      new Centroid(5, 1),
    ];
    expect<Centroid[]>(c1).toStrictEqual(c2);
  });
});

function logisticInvCdf(y: f64, mu: f64 = 0.0, s: f64 = 1.0): f64 {
  return mu - s * Math.log(1 / y - 1);
}

describe("mergeData", () => {
  it("total weight must be correct", () => {
    const N = 10000;
    const X = new Array<f64>(N)
      .fill(0)
      .map<f64>((_, i) => ((1.0 * i) / N) ** 2);
    // const C = centroidsFromPoints(X);
    const C = mergeData(centroidsFromPoints(X));
    const weight = C.reduce((x, c) => x + c.count, 0.0);

    // log(C);
    expect(weight).toBe(N);
  });

  it("correct approximate quantiles for logistic(0,1) distribution", () => {
    const N = 10000;
    const X = new Array<f64>(N)
      .fill(0)
      .map<f64>(() => Math.random())
      .map<f64>((x) => logisticInvCdf(x));
    // const C = centroidsFromPoints(X);
    log("length X " + X.length.toString());
    const C = mergeData(centroidsFromPoints(X), [], 50);
    const weight = C.reduce((x, c) => x + c.count, 0.0);

    log("length " + C.length.toString());
    log(C);
    expect(weight).toBe(N);
  });
});
