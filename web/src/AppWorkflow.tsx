import { useMemo, useState } from "react";
import Plot from "react-plotly.js";
import type { Config, Data, Layout } from "plotly.js";
import {
  applyAffine,
  diagnosticsFromAffine,
  invertAffine,
  solveAffine,
  type Affine,
  type AffineDiagnostics,
  type Point
} from "./affine";
import { parseDxfToPaths, transformPaths, writeSimpleDxfFromPaths, type Path2D } from "./dxf";

type PairRow = { ix: string; iy: string; mx: string; my: string };
type PickPoint = { x: number; y: number; label: string; source: string };
type DeltaRecord = { label: string; x: number; y: number; dx: number; dy: number };
type DxfLayer = "Ideal" | "Measured" | "Rectified" | "Predicted";

const defaultPairs: PairRow[] = [
  { ix: "0", iy: "0", mx: "0", my: "0" },
  { ix: "100", iy: "0", mx: "102", my: "1" },
  { ix: "0", iy: "100", mx: "-1", my: "98" }
];

function asNumber(value: string): number | null {
  if (value.trim() === "") return null;
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

function rowsToPoints(rows: PairRow[]) {
  const ideal: Point[] = [];
  const measured: Point[] = [];
  for (const r of rows) {
    const ix = asNumber(r.ix);
    const iy = asNumber(r.iy);
    const mx = asNumber(r.mx);
    const my = asNumber(r.my);
    if (ix === null || iy === null || mx === null || my === null) continue;
    ideal.push({ x: ix, y: iy });
    measured.push({ x: mx, y: my });
  }
  return { ideal, measured };
}

function closePath(points: Point[]): Path2D {
  if (points.length < 2) return points;
  return [...points, points[0]];
}

function tracesFromPaths(paths: Path2D[], color: string, name: string): Data[] {
  return paths.map((p, i) => ({
    x: p.map((q) => q.x),
    y: p.map((q) => q.y),
    type: "scatter",
    mode: "lines",
    line: { color, width: 2 },
    name: i === 0 ? name : undefined,
    showlegend: i === 0
  }));
}

function bboxCorners(points: Point[]): Point[] {
  if (points.length === 0) return [];
  const xs = points.map((p) => p.x);
  const ys = points.map((p) => p.y);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  return [
    { x: minX, y: minY },
    { x: minX, y: maxY },
    { x: maxX, y: maxY },
    { x: maxX, y: minY }
  ];
}

export default function AppWorkflow() {
  const [pairs, setPairs] = useState<PairRow[]>(defaultPairs);
  const [measuredCenterX, setMeasuredCenterX] = useState("");
  const [measuredCenterY, setMeasuredCenterY] = useState("");
  const [error, setError] = useState("");
  const [affine, setAffine] = useState<Affine | null>(null);
  const [diag, setDiag] = useState<AffineDiagnostics | null>(null);

  const [showIdeal, setShowIdeal] = useState(true);
  const [showMeasured, setShowMeasured] = useState(true);
  const [showRectified, setShowRectified] = useState(true);
  const [showPredicted, setShowPredicted] = useState(true);
  const [showCoords, setShowCoords] = useState(false);

  const [refPoint, setRefPoint] = useState<Point | null>(null);
  const [refX, setRefX] = useState("");
  const [refY, setRefY] = useState("");
  const [refMode, setRefMode] = useState(false);
  const [refRecords, setRefRecords] = useState<DeltaRecord[]>([]);

  const [validationPoints, setValidationPoints] = useState<PickPoint[]>([]);
  const [vIx, setVIx] = useState("");
  const [vIy, setVIy] = useState("");
  const [vMx, setVMx] = useState("");
  const [vMy, setVMy] = useState("");
  const [valIdealOut, setValIdealOut] = useState("");
  const [valMeasuredOut, setValMeasuredOut] = useState("");

  const [dxfIdealPaths, setDxfIdealPaths] = useState<Path2D[]>([]);
  const [showDxfIdeal, setShowDxfIdeal] = useState(true);
  const [showDxfMeasured, setShowDxfMeasured] = useState(true);
  const [showDxfRectified, setShowDxfRectified] = useState(true);
  const [showDxfPredicted, setShowDxfPredicted] = useState(true);
  const [showDxfCoords, setShowDxfCoords] = useState(false);
  const [dxfRefPoint, setDxfRefPoint] = useState<Point | null>(null);
  const [dxfRefX, setDxfRefX] = useState("");
  const [dxfRefY, setDxfRefY] = useState("");
  const [dxfRefMode, setDxfRefMode] = useState(false);
  const [dxfRefRecords, setDxfRefRecords] = useState<DeltaRecord[]>([]);
  const [dxfMeasureMode, setDxfMeasureMode] = useState(false);
  const [dxfMeasurePending, setDxfMeasurePending] = useState<PickPoint | null>(null);
  const [dxfMeasureRecords, setDxfMeasureRecords] = useState<{ a: PickPoint; b: PickPoint; dx: number; dy: number }[]>([]);
  const [dxfBboxOn, setDxfBboxOn] = useState(false);
  const [dxfBboxLayer, setDxfBboxLayer] = useState<DxfLayer>("Ideal");

  const points = useMemo(() => rowsToPoints(pairs), [pairs]);
  const rectifiedPoints = useMemo(() => {
    if (!affine) return [] as Point[];
    const inv = invertAffine(affine);
    return points.measured.map((p) => applyAffine(p, inv));
  }, [affine, points]);
  const predictedPoints = useMemo(() => {
    const n = Math.min(points.measured.length, rectifiedPoints.length);
    if (n === 0) return [] as Point[];
    return Array.from({ length: n }, (_, i) => ({
      x: 0.5 * (points.measured[i].x + rectifiedPoints[i].x),
      y: 0.5 * (points.measured[i].y + rectifiedPoints[i].y)
    }));
  }, [points.measured, rectifiedPoints]);

  const dxfMeasured = useMemo(() => (affine ? transformPaths(dxfIdealPaths, affine) : []), [dxfIdealPaths, affine]);
  const dxfRectified = useMemo(() => (affine ? transformPaths(dxfIdealPaths, invertAffine(affine)) : []), [dxfIdealPaths, affine]);
  const dxfPredicted = useMemo(() => (affine ? transformPaths(dxfRectified, affine) : []), [dxfRectified, affine]);

  const calibrationPickPoints = useMemo(() => {
    const out: PickPoint[] = [];
    const push = (name: string, pts: Point[]) => pts.forEach((p, i) => out.push({ x: p.x, y: p.y, label: `${name} #${i + 1}`, source: name }));
    if (showIdeal) push("Ideal", points.ideal);
    if (showMeasured) push("Measured", points.measured);
    if (showRectified) push("Rectified", rectifiedPoints);
    if (showPredicted) push("Predicted", predictedPoints);
    validationPoints.forEach((v) => out.push(v));
    return out;
  }, [showIdeal, showMeasured, showRectified, showPredicted, points, rectifiedPoints, predictedPoints, validationPoints]);

  const dxfLayers = useMemo(
    () => ({
      Ideal: showDxfIdeal ? dxfIdealPaths : [],
      Measured: showDxfMeasured ? dxfMeasured : [],
      Rectified: showDxfRectified ? dxfRectified : [],
      Predicted: showDxfPredicted ? dxfPredicted : []
    }),
    [showDxfIdeal, showDxfMeasured, showDxfRectified, showDxfPredicted, dxfIdealPaths, dxfMeasured, dxfRectified, dxfPredicted]
  );

  const dxfBbox = useMemo(() => {
    if (!dxfBboxOn) return [] as PickPoint[];
    const corners = bboxCorners(dxfLayers[dxfBboxLayer].flat());
    return corners.map((p, i) => ({ x: p.x, y: p.y, label: `BBox ${dxfBboxLayer} C${i + 1}`, source: `BBox ${dxfBboxLayer}` }));
  }, [dxfBboxOn, dxfBboxLayer, dxfLayers]);

  const dxfPickPoints = useMemo(() => {
    const out: PickPoint[] = [];
    (Object.entries(dxfLayers) as [DxfLayer, Path2D[]][]).forEach(([layer, paths]) =>
      paths.forEach((path, pi) => path.forEach((p, i) => out.push({ x: p.x, y: p.y, label: `${layer} p${pi + 1}:${i + 1}`, source: layer })))
    );
    out.push(...dxfBbox);
    return out;
  }, [dxfLayers, dxfBbox]);

  const nearest = (x: number, y: number, list: PickPoint[]) => {
    let best: PickPoint | null = null;
    let d2 = Number.POSITIVE_INFINITY;
    for (const p of list) {
      const dd = (x - p.x) ** 2 + (y - p.y) ** 2;
      if (dd < d2) {
        d2 = dd;
        best = p;
      }
    }
    return best;
  };

  const onSolve = () => {
    try {
      if (points.ideal.length < 3) throw new Error("At least 3 complete coordinate pairs are required.");
      const solved = solveAffine(points.ideal, points.measured);
      setAffine(solved);
      setDiag(diagnosticsFromAffine(solved, points.ideal, points.measured));
      setError("");
    } catch (e: any) {
      setError(e?.message || String(e));
    }
  };

  const onSetReference = () => {
    const x = asNumber(refX);
    const y = asNumber(refY);
    if (x === null || y === null) return;
    setRefPoint({ x, y });
    setRefRecords([]);
  };

  const calibrationPlotData: Data[] = useMemo(() => {
    const out: Data[] = [];
    if (showIdeal) out.push(...tracesFromPaths([closePath(points.ideal)], "#1d4ed8", "Ideal"));
    if (showMeasured) out.push(...tracesFromPaths([closePath(points.measured)], "#dc2626", "Measured"));
    if (showRectified) out.push(...tracesFromPaths([closePath(rectifiedPoints)], "#16a34a", "Rectified"));
    if (showPredicted) out.push(...tracesFromPaths([closePath(predictedPoints)], "#7c3aed", "Predicted"));

    if (showCoords) {
      const textTrace = (name: string, pts: Point[]): Data => ({
        x: pts.map((p) => p.x),
        y: pts.map((p) => p.y),
        type: "scatter",
        mode: "text",
        text: pts.map((p) => `${name} [${p.x.toFixed(3)}, ${p.y.toFixed(3)}]`),
        textposition: "top right",
        textfont: { size: 9 },
        showlegend: false,
        hoverinfo: "skip"
      });
      if (showIdeal) out.push(textTrace("Ideal", points.ideal));
      if (showMeasured) out.push(textTrace("Measured", points.measured));
      if (showRectified) out.push(textTrace("Rectified", rectifiedPoints));
      if (showPredicted) out.push(textTrace("Predicted", predictedPoints));
    }
    if (refPoint) {
      out.push({ x: [refPoint.x], y: [refPoint.y], type: "scatter", mode: "text+markers", marker: { color: "#111", symbol: "x", size: 11 }, text: ["Ref"], textposition: "top right", showlegend: false });
      refRecords.forEach((r) => {
        out.push({ x: [refPoint.x, r.x], y: [refPoint.y, r.y], type: "scatter", mode: "lines", line: { color: "#555", width: 1.2 }, showlegend: false });
        out.push({ x: [refPoint.x, r.x, r.x], y: [refPoint.y, refPoint.y, r.y], type: "scatter", mode: "text+lines", line: { color: "#666", dash: "dot", width: 1 }, text: ["", "", `DX=${r.dx.toFixed(3)}, DY=${r.dy.toFixed(3)}`], textposition: "top right", showlegend: false });
      });
    }
    if (validationPoints.length > 0) {
      out.push({ x: validationPoints.map((p) => p.x), y: validationPoints.map((p) => p.y), type: "scatter", mode: "text+markers", marker: { color: "#0f172a", size: 8, symbol: "diamond" }, text: validationPoints.map((p) => p.label), textposition: "bottom right", showlegend: false });
    }
    return out;
  }, [showIdeal, showMeasured, showRectified, showPredicted, showCoords, points, rectifiedPoints, predictedPoints, refPoint, refRecords, validationPoints]);

  const dxfPlotData: Data[] = useMemo(() => {
    const out: Data[] = [];
    if (showDxfIdeal) out.push(...tracesFromPaths(dxfIdealPaths, "#1d4ed8", "Ideal"));
    if (showDxfMeasured) out.push(...tracesFromPaths(dxfMeasured, "#dc2626", "Measured"));
    if (showDxfRectified) out.push(...tracesFromPaths(dxfRectified, "#16a34a", "Rectified"));
    if (showDxfPredicted) out.push(...tracesFromPaths(dxfPredicted, "#7c3aed", "Predicted"));
    if (showDxfCoords) {
      (Object.entries(dxfLayers) as [DxfLayer, Path2D[]][]).forEach(([layer, paths]) => {
        const pts = paths.flat();
        out.push({ x: pts.map((p) => p.x), y: pts.map((p) => p.y), type: "scatter", mode: "text", text: pts.map((p) => `${layer} [${p.x.toFixed(3)}, ${p.y.toFixed(3)}]`), textposition: "top right", textfont: { size: 9 }, showlegend: false, hoverinfo: "skip" });
      });
    }
    if (dxfBbox.length === 4) {
      const poly = [...dxfBbox, dxfBbox[0]];
      out.push({ x: poly.map((p) => p.x), y: poly.map((p) => p.y), type: "scatter", mode: "lines+markers", line: { color: "#cc6a00", dash: "dot", width: 1.3 }, marker: { color: "#cc6a00", symbol: "square", size: 7 }, name: "Bounding box", showlegend: true });
    }
    if (dxfRefPoint) {
      out.push({ x: [dxfRefPoint.x], y: [dxfRefPoint.y], type: "scatter", mode: "text+markers", marker: { color: "#111", symbol: "x", size: 11 }, text: ["Ref"], textposition: "top right", showlegend: false });
      dxfRefRecords.forEach((r) => {
        out.push({ x: [dxfRefPoint.x, r.x], y: [dxfRefPoint.y, r.y], type: "scatter", mode: "lines", line: { color: "#555", width: 1.2 }, showlegend: false });
        out.push({ x: [dxfRefPoint.x, r.x, r.x], y: [dxfRefPoint.y, dxfRefPoint.y, r.y], type: "scatter", mode: "text+lines", line: { color: "#666", width: 1, dash: "dot" }, text: ["", "", `DX=${r.dx.toFixed(3)}, DY=${r.dy.toFixed(3)}`], textposition: "top right", showlegend: false });
      });
    }
    if (dxfMeasurePending) {
      out.push({
        x: [dxfMeasurePending.x],
        y: [dxfMeasurePending.y],
        type: "scatter",
        mode: "text+markers",
        marker: { color: "#111", symbol: "x", size: 11 },
        text: ["A"],
        textposition: "top right",
        showlegend: false
      });
    }
    dxfMeasureRecords.forEach((r) => {
      out.push({
        x: [r.a.x, r.b.x],
        y: [r.a.y, r.b.y],
        type: "scatter",
        mode: "text+lines+markers",
        line: { color: "#333", width: 1.3 },
        marker: { color: "#111", size: 7 },
        text: ["", `DX=${r.dx.toFixed(3)}, DY=${r.dy.toFixed(3)}`],
        textposition: "top right",
        showlegend: false
      });
      out.push({
        x: [r.a.x, r.b.x, r.b.x],
        y: [r.a.y, r.a.y, r.b.y],
        type: "scatter",
        mode: "lines",
        line: { color: "#666", width: 1, dash: "dot" },
        showlegend: false
      });
    });
    out.push({ x: dxfPickPoints.map((p) => p.x), y: dxfPickPoints.map((p) => p.y), type: "scatter", mode: "markers", marker: { size: 12, color: "rgba(0,0,0,0)" }, text: dxfPickPoints.map((p) => p.label), hovertemplate: "%{text}<extra></extra>", showlegend: false });
    return out;
  }, [showDxfIdeal, showDxfMeasured, showDxfRectified, showDxfPredicted, showDxfCoords, dxfIdealPaths, dxfMeasured, dxfRectified, dxfPredicted, dxfLayers, dxfBbox, dxfRefPoint, dxfRefRecords, dxfMeasurePending, dxfMeasureRecords, dxfPickPoints]);

  const layout: Partial<Layout> = {
    autosize: true,
    height: 540,
    paper_bgcolor: "white",
    plot_bgcolor: "white",
    xaxis: { title: { text: "X" } },
    yaxis: { title: { text: "Y" }, scaleanchor: "x" }
  };
  const config: Partial<Config> = { responsive: true };

  return (
    <div className="page">
      <section className="hero card">
        <h1>Machine DXF Affine Calibrator</h1>
        <p>Same workflow as desktop app with a modern, cleaner web layout.</p>
      </section>

      <section className="card">
        <h2>1) Calibration Inputs</h2>
        <p className="sub">Ideal center is fixed at 0,0. Measured center is optional. Minimum 3 complete coordinate pairs required.</p>
        <div className="grid center-grid">
          <label>Ideal center X</label>
          <input type="number" value="0" readOnly />
          <label>Ideal center Y</label>
          <input type="number" value="0" readOnly />
          <label>Measured center X (optional)</label>
          <input type="number" value={measuredCenterX} onChange={(e) => setMeasuredCenterX(e.target.value)} />
          <label>Measured center Y (optional)</label>
          <input type="number" value={measuredCenterY} onChange={(e) => setMeasuredCenterY(e.target.value)} />
        </div>
        <div className="grid head"><b>Ideal X</b><b>Ideal Y</b><b>Measured X</b><b>Measured Y</b></div>
        {pairs.map((p, i) => (
          <div className="grid row" key={i}>
            <input type="number" value={p.ix} onChange={(e) => setPairs((old) => old.map((x, j) => j === i ? { ...x, ix: e.target.value } : x))} />
            <input type="number" value={p.iy} onChange={(e) => setPairs((old) => old.map((x, j) => j === i ? { ...x, iy: e.target.value } : x))} />
            <input type="number" value={p.mx} onChange={(e) => setPairs((old) => old.map((x, j) => j === i ? { ...x, mx: e.target.value } : x))} />
            <input type="number" value={p.my} onChange={(e) => setPairs((old) => old.map((x, j) => j === i ? { ...x, my: e.target.value } : x))} />
          </div>
        ))}
        <div className="actions">
          <button onClick={() => setPairs((old) => [...old, { ix: "", iy: "", mx: "", my: "" }])}>Add coordinate pair</button>
          <button onClick={() => setPairs((old) => (old.length > 3 ? old.slice(0, -1) : old))}>Remove last pair</button>
          <button onClick={onSolve}>Solve Calibration</button>
          <button onClick={() => {
            const lines = [
              "type,ideal_x,ideal_y,measured_x,measured_y",
              `center,0,0,${measuredCenterX},${measuredCenterY}`,
              ...pairs.map((p) => `pair,${p.ix},${p.iy},${p.mx},${p.my}`)
            ];
            const blob = new Blob([lines.join("\n")], { type: "text/csv" });
            const a = document.createElement("a");
            a.href = URL.createObjectURL(blob);
            a.download = "calibration_inputs.csv";
            a.click();
            URL.revokeObjectURL(a.href);
          }}>Save CSV preset</button>
          <label className="file-label">Load CSV preset<input type="file" accept=".csv" onChange={async (e) => {
            const f = e.target.files?.[0];
            if (!f) return;
            const text = await f.text();
            const rows = text.split(/\r?\n/).filter(Boolean);
            if (rows.length === 0) {
              setError("CSV is empty.");
              return;
            }
            const header = rows[0].split(",").map((x) => x.trim().toLowerCase());
            const required = ["type", "ideal_x", "ideal_y", "measured_x", "measured_y"];
            if (required.some((k) => !header.includes(k))) {
              setError("CSV header must include: type,ideal_x,ideal_y,measured_x,measured_y");
              return;
            }
            const out: PairRow[] = [];
            let centerCount = 0;
            for (let i = 1; i < rows.length; i++) {
              const c = rows[i].split(",");
              if (c[0] === "center") {
                centerCount += 1;
                const ix = (c[1] ?? "").trim();
                const iy = (c[2] ?? "").trim();
                if ((ix !== "" && ix !== "0") || (iy !== "" && iy !== "0")) {
                  setError("CSV center row must keep ideal center as 0,0.");
                  return;
                }
                setMeasuredCenterX(c[3] ?? "");
                setMeasuredCenterY(c[4] ?? "");
              } else if (c[0] === "pair") {
                out.push({ ix: c[1] ?? "", iy: c[2] ?? "", mx: c[3] ?? "", my: c[4] ?? "" });
              }
            }
            if (centerCount !== 1) {
              setError("CSV must contain exactly one center row.");
              return;
            }
            if (out.length > 0) setPairs(out);
            setError("");
          }} /></label>
        </div>
      </section>

      <section className="card">
        <h2>2) Calibration Diagnostics</h2>
        {diag ? (
          <div className="diag-grid">
            <div>Translation X: <b>{diag.translationX.toFixed(6)}</b></div>
            <div>Translation Y: <b>{diag.translationY.toFixed(6)}</b></div>
            <div>Rotation (deg): <b>{diag.rotationDeg.toFixed(6)}</b></div>
            <div>Scale X: <b>{diag.scaleX.toFixed(6)}</b></div>
            <div>Scale Y: <b>{diag.scaleY.toFixed(6)}</b></div>
            <div>Shear: <b>{diag.shear.toFixed(6)}</b></div>
            <div>RMS Error: <b>{diag.rmsError.toFixed(6)}</b></div>
          </div>
        ) : <p className="sub">Run Solve Calibration to show diagnostics.</p>}
      </section>

      <section className="card">
        <h2>3) Calibration Preview + Reference</h2>
        <div className="actions">
          <label><input type="checkbox" checked={showIdeal} onChange={(e) => setShowIdeal(e.target.checked)} /> Ideal</label>
          <label><input type="checkbox" checked={showMeasured} onChange={(e) => setShowMeasured(e.target.checked)} /> Measured</label>
          <label><input type="checkbox" checked={showRectified} onChange={(e) => setShowRectified(e.target.checked)} /> Rectified</label>
          <label><input type="checkbox" checked={showPredicted} onChange={(e) => setShowPredicted(e.target.checked)} /> Predicted</label>
          <label><input type="checkbox" checked={showCoords} onChange={(e) => setShowCoords(e.target.checked)} /> Coordinates</label>
        </div>
        <div className="actions">
          <input type="number" placeholder="Reference X" value={refX} onChange={(e) => setRefX(e.target.value)} />
          <input type="number" placeholder="Reference Y" value={refY} onChange={(e) => setRefY(e.target.value)} />
          <button onClick={onSetReference}>Set reference</button>
          <button disabled={!refPoint} onClick={() => setRefMode((v) => !v)}>{refMode ? "Stop dimensions" : "Add dimensions"}</button>
          <button onClick={() => setRefRecords([])}>Clear dimensions</button>
          <button onClick={() => { setRefPoint(null); setRefMode(false); setRefRecords([]); }}>Clear reference</button>
        </div>
        <Plot
          data={calibrationPlotData}
          layout={layout}
          style={{ width: "100%" }}
          config={config}
          onClick={(event) => {
            const p = event?.points?.[0];
            if (!p || !refMode || !refPoint) return;
            const hit = nearest(Number(p.x), Number(p.y), calibrationPickPoints);
            if (!hit) return;
            setRefRecords((old) => [...old, { label: hit.label, x: hit.x, y: hit.y, dx: hit.x - refPoint.x, dy: hit.y - refPoint.y }]);
          }}
        />
        <div className="log">
          {refRecords.length === 0 ? "Reference dimension log is empty." : refRecords.map((r, i) => <div key={i}>#{i + 1} {r.label}: DX={r.dx.toFixed(3)}, DY={r.dy.toFixed(3)}</div>)}
        </div>
      </section>

      <section className="card">
        <h2>4) Validation</h2>
        <div className="split">
          <div>
            <h3>Ideal → Measured</h3>
            <div className="actions">
              <input type="number" placeholder="Ideal X" value={vIx} onChange={(e) => setVIx(e.target.value)} />
              <input type="number" placeholder="Ideal Y" value={vIy} onChange={(e) => setVIy(e.target.value)} />
              <button disabled={!affine} onClick={() => {
                if (!affine) return;
                const x = asNumber(vIx); const y = asNumber(vIy);
                if (x === null || y === null) return;
                const out = applyAffine({ x, y }, affine);
                setValIdealOut(`[${out.x.toFixed(4)}, ${out.y.toFixed(4)}]`);
                setValidationPoints((old) => [...old, { x, y, label: "Validation Ideal In", source: "Validation" }, { x: out.x, y: out.y, label: "Validation Measured Out", source: "Validation" }]);
              }}>Verify</button>
            </div>
            <div className="out">{valIdealOut || "Output appears here."}</div>
          </div>
          <div>
            <h3>Measured → Ideal</h3>
            <div className="actions">
              <input type="number" placeholder="Measured X" value={vMx} onChange={(e) => setVMx(e.target.value)} />
              <input type="number" placeholder="Measured Y" value={vMy} onChange={(e) => setVMy(e.target.value)} />
              <button disabled={!affine} onClick={() => {
                if (!affine) return;
                const x = asNumber(vMx); const y = asNumber(vMy);
                if (x === null || y === null) return;
                const out = applyAffine({ x, y }, invertAffine(affine));
                setValMeasuredOut(`[${out.x.toFixed(4)}, ${out.y.toFixed(4)}]`);
                setValidationPoints((old) => [...old, { x, y, label: "Validation Measured In", source: "Validation" }, { x: out.x, y: out.y, label: "Validation Ideal Out", source: "Validation" }]);
              }}>Verify</button>
            </div>
            <div className="out">{valMeasuredOut || "Output appears here."}</div>
          </div>
        </div>
      </section>

      <section className="card">
        <h2>5) DXF Input / Output</h2>
        <input type="file" accept=".dxf" onChange={async (e) => {
          const f = e.target.files?.[0];
          if (!f) return;
          try {
            setDxfIdealPaths(parseDxfToPaths(await f.text()));
            setError("");
          } catch (err: any) {
            setError(`DXF parse failed: ${err?.message || err}`);
          }
        }} />
        <div className="actions">
          <button disabled={!affine || dxfRectified.length === 0} onClick={() => {
            const dxf = writeSimpleDxfFromPaths(dxfRectified);
            const blob = new Blob([dxf], { type: "application/dxf" });
            const a = document.createElement("a");
            a.href = URL.createObjectURL(blob);
            a.download = "rectified_output.dxf";
            a.click();
            URL.revokeObjectURL(a.href);
          }}>Download Rectified DXF</button>
        </div>
      </section>

      <section className="card">
        <h2>6) DXF Comparison + Reference</h2>
        <div className="actions">
          <label><input type="checkbox" checked={showDxfIdeal} onChange={(e) => setShowDxfIdeal(e.target.checked)} /> Ideal</label>
          <label><input type="checkbox" checked={showDxfMeasured} onChange={(e) => setShowDxfMeasured(e.target.checked)} /> Measured</label>
          <label><input type="checkbox" checked={showDxfRectified} onChange={(e) => setShowDxfRectified(e.target.checked)} /> Rectified</label>
          <label><input type="checkbox" checked={showDxfPredicted} onChange={(e) => setShowDxfPredicted(e.target.checked)} /> Predicted</label>
          <label><input type="checkbox" checked={showDxfCoords} onChange={(e) => setShowDxfCoords(e.target.checked)} /> Coordinates</label>
          <label><input type="checkbox" checked={dxfBboxOn} onChange={(e) => setDxfBboxOn(e.target.checked)} /> Bounding box</label>
          <label>Layer
            <select value={dxfBboxLayer} onChange={(e) => setDxfBboxLayer(e.target.value as DxfLayer)}>
              <option>Ideal</option>
              <option>Measured</option>
              <option>Rectified</option>
              <option>Predicted</option>
            </select>
          </label>
        </div>
        <div className="actions">
          <input type="number" placeholder="Reference X" value={dxfRefX} onChange={(e) => setDxfRefX(e.target.value)} />
          <input type="number" placeholder="Reference Y" value={dxfRefY} onChange={(e) => setDxfRefY(e.target.value)} />
          <button onClick={() => {
            const x = asNumber(dxfRefX);
            const y = asNumber(dxfRefY);
            if (x === null || y === null) return;
            setDxfRefPoint({ x, y });
            setDxfRefRecords([]);
          }}>Set reference</button>
          <button disabled={!dxfRefPoint} onClick={() => {
            setDxfRefMode((v) => !v);
            setDxfMeasureMode(false);
            setDxfMeasurePending(null);
          }}>{dxfRefMode ? "Stop dimensions" : "Add dimensions"}</button>
          <button onClick={() => setDxfRefRecords([])}>Clear dimensions</button>
          <button onClick={() => { setDxfRefPoint(null); setDxfRefMode(false); setDxfRefRecords([]); }}>Clear reference</button>
          <button onClick={() => {
            setDxfMeasureMode((v) => !v);
            setDxfRefMode(false);
            setDxfMeasurePending(null);
          }}>{dxfMeasureMode ? "Stop measure" : "Measure DX/DY"}</button>
          <button onClick={() => { setDxfMeasureRecords([]); setDxfMeasurePending(null); }}>Clear measurements</button>
        </div>
        <Plot
          data={dxfPlotData}
          layout={layout}
          style={{ width: "100%" }}
          config={config}
          onClick={(event) => {
            const p = event?.points?.[0];
            if (!p) return;
            const hit = nearest(Number(p.x), Number(p.y), dxfPickPoints);
            if (!hit) return;
            if (dxfRefMode && dxfRefPoint) {
              setDxfRefRecords((old) => [...old, { label: hit.label, x: hit.x, y: hit.y, dx: hit.x - dxfRefPoint.x, dy: hit.y - dxfRefPoint.y }]);
              return;
            }
            if (!dxfMeasureMode) return;
            if (!dxfMeasurePending) {
              setDxfMeasurePending(hit);
              return;
            }
            setDxfMeasureRecords((old) => [...old, { a: dxfMeasurePending, b: hit, dx: hit.x - dxfMeasurePending.x, dy: hit.y - dxfMeasurePending.y }]);
            setDxfMeasurePending(null);
          }}
        />
        <div className="split">
          <div className="log">
            {dxfRefRecords.length === 0 ? "DXF reference log is empty." : dxfRefRecords.map((r, i) => <div key={i}>#{i + 1} {r.label}: DX={r.dx.toFixed(3)}, DY={r.dy.toFixed(3)}</div>)}
          </div>
          <div className="log">
            {dxfMeasureRecords.length === 0 ? "DXF measurement log is empty." : dxfMeasureRecords.map((r, i) => <div key={i}>#{i + 1} {r.a.label} → {r.b.label}: DX={r.dx.toFixed(3)}, DY={r.dy.toFixed(3)}</div>)}
          </div>
        </div>
      </section>

      {error && <section className="card error">{error}</section>}
    </div>
  );
}
