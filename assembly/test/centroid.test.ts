import { Centroid } from "../centroid";

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
