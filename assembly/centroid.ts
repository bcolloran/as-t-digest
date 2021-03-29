export class Centroid {
  constructor(public mean: f64, public count: f64) {}

  @operator("+")
  static add(a: Centroid, b: Centroid): Centroid {
    const count = a.count + b.count;
    const mean = (a.count * a.mean + b.count * b.mean) / count;
    return new Centroid(mean, count);
  }

  updateAdd(a: Centroid): void {
    const count = a.count + this.count;
    const mean = (a.count * a.mean + this.count * this.mean) / count;
    this.mean = mean;
    this.count = count;
  }
}

export function centroidSortFn(c1: Centroid, c2: Centroid): i32 {
  return c1.mean < c2.mean ? -1 : c1.mean === c2.mean ? 0 : 1;
}
