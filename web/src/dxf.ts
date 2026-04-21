import DxfParser from "dxf-parser";
import { applyAffine, type Affine, type Point } from "./affine";

export type Path2D = Point[];

export function parseDxfToPaths(content: string): Path2D[] {
  const parser = new DxfParser();
  const doc: any = parser.parseSync(content);
  const out: Path2D[] = [];
  for (const e of doc.entities ?? []) {
    if (e.type === "LINE") {
      out.push([{ x: e.vertices[0].x, y: e.vertices[0].y }, { x: e.vertices[1].x, y: e.vertices[1].y }]);
    } else if (e.type === "LWPOLYLINE" || e.type === "POLYLINE") {
      const pts = (e.vertices ?? []).map((v: any) => ({ x: v.x, y: v.y }));
      if (pts.length > 1) out.push(pts);
    }
  }
  return out;
}

export function transformPaths(paths: Path2D[], t: Affine): Path2D[] {
  return paths.map((p) => p.map((q) => applyAffine(q, t)));
}

export function writeSimpleDxfFromPaths(paths: Path2D[]): string {
  const lines: string[] = ["0", "SECTION", "2", "ENTITIES"];
  for (const path of paths) {
    if (path.length < 2) continue;
    lines.push("0", "LWPOLYLINE", "100", "AcDbEntity", "8", "0", "100", "AcDbPolyline", "90", String(path.length), "70", "0");
    for (const p of path) {
      lines.push("10", String(p.x), "20", String(p.y));
    }
  }
  lines.push("0", "ENDSEC", "0", "EOF");
  return lines.join("\n");
}
