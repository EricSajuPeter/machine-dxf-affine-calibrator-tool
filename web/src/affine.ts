export type Point = { x: number; y: number };

export type Affine = {
  a11: number;
  a12: number;
  a21: number;
  a22: number;
  tx: number;
  ty: number;
};

export type AffineDiagnostics = {
  translationX: number;
  translationY: number;
  rotationDeg: number;
  scaleX: number;
  scaleY: number;
  shear: number;
  rmsError: number;
};

function solveLinear6(a: number[][], b: number[]): number[] {
  const n = 6;
  for (let i = 0; i < n; i++) {
    let pivot = i;
    for (let r = i + 1; r < n; r++) if (Math.abs(a[r][i]) > Math.abs(a[pivot][i])) pivot = r;
    [a[i], a[pivot]] = [a[pivot], a[i]];
    [b[i], b[pivot]] = [b[pivot], b[i]];
    const div = a[i][i] || 1e-12;
    for (let c = i; c < n; c++) a[i][c] /= div;
    b[i] /= div;
    for (let r = 0; r < n; r++) {
      if (r === i) continue;
      const f = a[r][i];
      for (let c = i; c < n; c++) a[r][c] -= f * a[i][c];
      b[r] -= f * b[i];
    }
  }
  return b;
}

export function solveAffine(ideal: Point[], measured: Point[]): Affine {
  if (ideal.length < 3 || measured.length !== ideal.length) {
    throw new Error("Need >= 3 matching point pairs.");
  }
  const rows: number[][] = [];
  const rhs: number[] = [];
  for (let i = 0; i < ideal.length; i++) {
    const p = ideal[i];
    const q = measured[i];
    rows.push([p.x, p.y, 0, 0, 1, 0]); rhs.push(q.x);
    rows.push([0, 0, p.x, p.y, 0, 1]); rhs.push(q.y);
  }
  const ata = Array.from({ length: 6 }, () => Array(6).fill(0));
  const atb = Array(6).fill(0);
  for (let r = 0; r < rows.length; r++) {
    for (let i = 0; i < 6; i++) {
      atb[i] += rows[r][i] * rhs[r];
      for (let j = 0; j < 6; j++) ata[i][j] += rows[r][i] * rows[r][j];
    }
  }
  const [a11, a12, a21, a22, tx, ty] = solveLinear6(ata, atb);
  return { a11, a12, a21, a22, tx, ty };
}

export function applyAffine(p: Point, t: Affine): Point {
  return { x: t.a11 * p.x + t.a12 * p.y + t.tx, y: t.a21 * p.x + t.a22 * p.y + t.ty };
}

export function invertAffine(t: Affine): Affine {
  const det = t.a11 * t.a22 - t.a12 * t.a21;
  if (Math.abs(det) < 1e-12) throw new Error("Affine matrix is singular.");
  const ia11 = t.a22 / det;
  const ia12 = -t.a12 / det;
  const ia21 = -t.a21 / det;
  const ia22 = t.a11 / det;
  return {
    a11: ia11,
    a12: ia12,
    a21: ia21,
    a22: ia22,
    tx: -(ia11 * t.tx + ia12 * t.ty),
    ty: -(ia21 * t.tx + ia22 * t.ty)
  };
}

export function diagnosticsFromAffine(
  transform: Affine,
  ideal: Point[],
  measured: Point[]
): AffineDiagnostics {
  const tx = transform.tx;
  const ty = transform.ty;
  const rotation = Math.atan2(transform.a21, transform.a11) * (180 / Math.PI);
  const scaleX = Math.hypot(transform.a11, transform.a21);
  const scaleY = Math.hypot(transform.a12, transform.a22);
  const shear =
    scaleX > 1e-12 ? (transform.a11 * transform.a12 + transform.a21 * transform.a22) / (scaleX * scaleX) : 0;

  let sumSq = 0;
  const n = Math.min(ideal.length, measured.length);
  for (let i = 0; i < n; i++) {
    const pred = applyAffine(ideal[i], transform);
    const dx = pred.x - measured[i].x;
    const dy = pred.y - measured[i].y;
    sumSq += dx * dx + dy * dy;
  }
  const rms = n > 0 ? Math.sqrt(sumSq / n) : 0;

  return {
    translationX: tx,
    translationY: ty,
    rotationDeg: rotation,
    scaleX,
    scaleY,
    shear,
    rmsError: rms
  };
}
