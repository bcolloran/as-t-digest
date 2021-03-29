import { Centroid } from "../centroid";
import { centroidsFromPoints, findClosestCentroids } from "../utils";

describe("centroidsFromPoints", () => {
  xit("centroidsFromPoints works correctly", () => {
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

describe("findClosestCentroids", () => {
  xit("empty list of centroids", () => {
    const x = new Centroid(0, 1);
    const c: Centroid[] = [];

    expect<Centroid[]>(findClosestCentroids(x, c)).toStrictEqual([]);
  });

  xit("bottommost single centroid", () => {
    const x = new Centroid(0, 1);
    const c = [
      new Centroid(2, 1),
      new Centroid(4, 1),
      new Centroid(6, 1),
      new Centroid(8, 1),
    ];

    expect<Centroid[]>(findClosestCentroids(x, c)).toStrictEqual([
      new Centroid(2, 1),
    ]);
  });

  xit("topmost single centroid", () => {
    const x = new Centroid(9, 1);
    const c = [
      new Centroid(2, 1),
      new Centroid(4, 1),
      new Centroid(6, 1),
      new Centroid(8, 1),
    ];

    expect<Centroid[]>(findClosestCentroids(x, c)).toStrictEqual([
      new Centroid(8, 1),
    ]);
  });

  xit("midpoint two centroids", () => {
    const x = new Centroid(5, 1);
    const c = [
      new Centroid(2, 1),
      new Centroid(4, 1),
      new Centroid(6, 1),
      new Centroid(8, 1),
    ];

    expect<Centroid[]>(findClosestCentroids(x, c)).toStrictEqual([
      new Centroid(4, 1),
      new Centroid(6, 1),
    ]);
  });

  xit("unique single closest centroid", () => {
    const x = new Centroid(4.1, 1);
    const c = [
      new Centroid(2, 1),
      new Centroid(4, 1),
      new Centroid(6, 1),
      new Centroid(8, 1),
    ];

    expect<Centroid[]>(findClosestCentroids(x, c)).toStrictEqual([
      new Centroid(4, 1),
    ]);
  });
});
