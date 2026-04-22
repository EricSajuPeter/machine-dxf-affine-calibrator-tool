import { useMemo, useState } from "react";
import Plot from "react-plotly.js";
import type { Config, Data, Layout } from "plotly.js";
import { invertAffine, solveAffine, type Affine, type Point } from "./affine";
import { parseDxfToPaths, transformPaths, writeSimpleDxfFromPaths, type Path2D } from "./dxf";

type Pair = { ix: number; iy: number; mx: number; my: number };

const defaultPairs: Pair[] = [
  { ix: 0, iy: 0, mx: 0, my: 0 },
  { ix: 100, iy: 0, mx: 102, my: 1 },
  { ix: 0, iy: 100, mx: -1, my: 98 }
];

type LayerName = "Input" | "Output" | "Distorted";
type PickPoint = {
  x: number;
  y: number;
  label: string;
  source: string;
};

type DeltaRecord = {
  label: string;
  x: number;
  y: number;
  dx: number;
  dy: number;
};

type MeasureRecord = {
  a: PickPoint;
  b: PickPoint;
  dx: number;
  dy: number;
};

function pathsToTraces(paths: Path2D[], color: string, name: string) {
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

export default function App() {
  const [pairs, setPairs] = useState<Pair[]>(defaultPairs);
  const [affine, setAffine] = useState<Affine | null>(null);
  const [error, setError] = useState("");
  const [inputPaths, setInputPaths] = useState<Path2D[]>([]);
  const [showInput, setShowInput] = useState(true);
  const [showOutput, setShowOutput] = useState(true);
  const [showDistorted, setShowDistorted] = useState(true);
  const [showCoords, setShowCoords] = useState(false);
  const [bboxEnabled, setBboxEnabled] = useState(false);
  const [bboxLayer, setBboxLayer] = useState<LayerName>("Input");
  const [measureMode, setMeasureMode] = useState(false);
  const [measurePending, setMeasurePending] = useState<PickPoint | null>(null);
  const [measureRecords, setMeasureRecords] = useState<MeasureRecord[]>([]);
  const [refX, setRefX] = useState("0");
  const [refY, setRefY] = useState("0");
  const [referencePoint, setReferencePoint] = useState<Point | null>(null);
  const [referenceMode, setReferenceMode] = useState(false);
  const [referenceRecords, setReferenceRecords] = useState<DeltaRecord[]>([]);

  const outputPaths = useMemo(() => {
    if (!affine) return [];
    const comp = invertAffine(affine);
    return transformPaths(inputPaths, comp);
  }, [inputPaths, affine]);

  const distortedPaths = useMemo(() => (affine ? transformPaths(inputPaths, affine) : []), [inputPaths, affine]);

  const visibleLayers = useMemo(() => {
    const map: Record<LayerName, Path2D[]> = {
      Input: showInput ? inputPaths : [],
      Output: showOutput ? outputPaths : [],
      Distorted: showDistorted ? distortedPaths : []
    };
    return map;
  }, [showInput, showOutput, showDistorted, inputPaths, outputPaths, distortedPaths]);

  const bboxCorners = useMemo(() => {
    if (!bboxEnabled) return [] as PickPoint[];
    const paths = visibleLayers[bboxLayer];
    const points = paths.flat();
    if (points.length === 0) return [] as PickPoint[];
    const xs = points.map((p) => p.x);
    const ys = points.map((p) => p.y);
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);
    return [
      { x: minX, y: minY, label: `BBox ${bboxLayer} C1`, source: `BBox ${bboxLayer}` },
      { x: minX, y: maxY, label: `BBox ${bboxLayer} C2`, source: `BBox ${bboxLayer}` },
      { x: maxX, y: maxY, label: `BBox ${bboxLayer} C3`, source: `BBox ${bboxLayer}` },
      { x: maxX, y: minY, label: `BBox ${bboxLayer} C4`, source: `BBox ${bboxLayer}` }
    ] as PickPoint[];
  }, [bboxEnabled, bboxLayer, visibleLayers]);

  const pickPoints = useMemo(() => {
    const out: PickPoint[] = [];
    const pushFrom = (layer: LayerName, paths: Path2D[]) => {
      for (let pi = 0; pi < paths.length; pi++) {
        const path = paths[pi];
        for (let i = 0; i < path.length; i++) {
          const p = path[i];
          out.push({ x: p.x, y: p.y, label: `${layer} p${pi + 1}:${i + 1}`, source: layer });
        }
      }
    };
    pushFrom("Input", visibleLayers.Input);
    pushFrom("Output", visibleLayers.Output);
    pushFrom("Distorted", visibleLayers.Distorted);
    out.push(...bboxCorners);
    return out;
  }, [visibleLayers, bboxCorners]);

  const traces = useMemo(() => {
    const t: Data[] = [];
    if (showInput) t.push(...(pathsToTraces(inputPaths, "#2c7be5", "Input") as Data[]));
    if (showOutput) t.push(...(pathsToTraces(outputPaths, "#00a86b", "Output") as Data[]));
    if (showDistorted) t.push(...(pathsToTraces(distortedPaths, "#8e44ad", "Distorted") as Data[]));

    if (bboxCorners.length === 4) {
      const poly = [...bboxCorners, bboxCorners[0]];
      t.push({
        x: poly.map((p) => p.x),
        y: poly.map((p) => p.y),
        type: "scatter",
        mode: "lines+markers",
        name: "Bounding box",
        line: { color: "#cc6a00", dash: "dot", width: 1.6 },
        marker: { color: "#cc6a00", size: 8, symbol: "square" },
        showlegend: true
      });
    }

    if (showCoords) {
      for (const layer of ["Input", "Output", "Distorted"] as const) {
        const pts = visibleLayers[layer].flat();
        if (pts.length === 0) continue;
        t.push({
          x: pts.map((p) => p.x),
          y: pts.map((p) => p.y),
          type: "scatter",
          mode: "text",
          text: pts.map((p) => `${layer} [${p.x.toFixed(3)}, ${p.y.toFixed(3)}]`),
          textposition: "top right",
          textfont: { size: 9 },
          hoverinfo: "skip",
          showlegend: false
        });
      }
      if (bboxCorners.length > 0) {
        t.push({
          x: bboxCorners.map((p) => p.x),
          y: bboxCorners.map((p) => p.y),
          type: "scatter",
          mode: "text",
          text: bboxCorners.map((p) => `[${p.x.toFixed(3)}, ${p.y.toFixed(3)}]`),
          textposition: "bottom right",
          textfont: { size: 9, color: "#8a4700" },
          hoverinfo: "skip",
          showlegend: false
        });
      }
    }

    if (referencePoint) {
      t.push({
        x: [referencePoint.x],
        y: [referencePoint.y],
        type: "scatter",
        mode: "text+markers",
        marker: { color: "black", symbol: "x", size: 12 },
        text: ["Ref"],
        textposition: "top right",
        showlegend: false
      });
      for (const rec of referenceRecords) {
        t.push({
          x: [referencePoint.x, rec.x],
          y: [referencePoint.y, rec.y],
          type: "scatter",
          mode: "lines",
          line: { color: "#555", width: 1.3 },
          showlegend: false
        });
        t.push({
          x: [referencePoint.x, rec.x, rec.x],
          y: [referencePoint.y, referencePoint.y, rec.y],
          type: "scatter",
          mode: "text+lines",
          line: { color: "#666", width: 1, dash: "dot" },
          text: ["", "", `DX=${rec.dx.toFixed(3)}, DY=${rec.dy.toFixed(3)}`],
          textposition: "top right",
          showlegend: false
        });
      }
    }

    if (measurePending) {
      t.push({
        x: [measurePending.x],
        y: [measurePending.y],
        type: "scatter",
        mode: "text+markers",
        marker: { color: "black", symbol: "x", size: 11 },
        text: ["A"],
        textposition: "top right",
        showlegend: false
      });
    }
    for (const rec of measureRecords) {
      t.push({
        x: [rec.a.x, rec.b.x],
        y: [rec.a.y, rec.b.y],
        type: "scatter",
        mode: "text+lines+markers",
        line: { color: "#333", width: 1.3 },
        marker: { color: "#111", size: 7 },
        text: ["", `DX=${rec.dx.toFixed(3)}, DY=${rec.dy.toFixed(3)}`],
        textposition: "top right",
        showlegend: false
      });
      t.push({
        x: [rec.a.x, rec.b.x, rec.b.x],
        y: [rec.a.y, rec.a.y, rec.b.y],
        type: "scatter",
        mode: "lines",
        line: { color: "#666", width: 1, dash: "dot" },
        showlegend: false
      });
    }

    if (pickPoints.length > 0) {
      t.push({
        x: pickPoints.map((p) => p.x),
        y: pickPoints.map((p) => p.y),
        type: "scatter",
        mode: "markers",
        marker: { size: 12, color: "rgba(0,0,0,0)" },
        text: pickPoints.map((p) => p.label),
        customdata: pickPoints.map((p) => `${p.label}|${p.source}`),
        hovertemplate: "%{text}<extra></extra>",
        showlegend: false
      });
    }

    return t;
  }, [
    showInput,
    showOutput,
    showDistorted,
    showCoords,
    inputPaths,
    outputPaths,
    distortedPaths,
    visibleLayers,
    bboxCorners,
    pickPoints,
    referencePoint,
    referenceRecords,
    measurePending,
    measureRecords
  ]);

  const plotData: Data[] = traces as Data[];
  const plotLayout: Partial<Layout> = {
    autosize: true,
    height: 560,
    paper_bgcolor: "white",
    plot_bgcolor: "white",
    xaxis: { title: { text: "X" } },
    yaxis: { title: { text: "Y" }, scaleanchor: "x" }
  };
  const plotConfig: Partial<Config> = { responsive: true };

  const onSolve = () => {
    try {
      setError("");
      const ideal: Point[] = pairs.map((p) => ({ x: p.ix, y: p.iy }));
      const measured: Point[] = pairs.map((p) => ({ x: p.mx, y: p.my }));
      setAffine(solveAffine(ideal, measured));
    } catch (e: any) {
      setError(e.message || String(e));
    }
  };

  const onUploadDxf = async (f: File) => {
    try {
      const text = await f.text();
      setInputPaths(parseDxfToPaths(text));
      setError("");
    } catch (e: any) {
      setError(`DXF parse failed: ${e.message || e}`);
    }
  };

  const onDownloadOutput = () => {
    const dxf = writeSimpleDxfFromPaths(outputPaths);
    const blob = new Blob([dxf], { type: "application/dxf" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "compensated_output.dxf";
    a.click();
    URL.revokeObjectURL(a.href);
  };

  const onSetReference = () => {
    const x = Number(refX);
    const y = Number(refY);
    if (!Number.isFinite(x) || !Number.isFinite(y)) {
      setError("Reference point must be numeric.");
      return;
    }
    setReferencePoint({ x, y });
    setReferenceRecords([]);
    setError("");
  };

  const onPlotClick = (event: any) => {
    const pt = event?.points?.[0];
    if (!pt) return;
    const label = (pt.text as string) || "point";
    const source = String(pt.customdata || "");
    const picked: PickPoint = { x: Number(pt.x), y: Number(pt.y), label, source };

    if (referenceMode && referencePoint) {
      const dx = picked.x - referencePoint.x;
      const dy = picked.y - referencePoint.y;
      setReferenceRecords((old) => [...old, { label: picked.label, x: picked.x, y: picked.y, dx, dy }]);
      return;
    }
    if (!measureMode) return;
    if (!measurePending) {
      setMeasurePending(picked);
      return;
    }
    const dx = picked.x - measurePending.x;
    const dy = picked.y - measurePending.y;
    setMeasureRecords((old) => [...old, { a: measurePending, b: picked, dx, dy }]);
    setMeasurePending(null);
  };

  return (
    <div className="page">
      <section className="hero card">
        <h1>Machine DXF Affine Calibrator</h1>
        <p>
          Browser-based calibration and DXF compensation. Upload your DXF, solve affine mismatch from matched points, preview layers, and download compensated output.
          Files stay on your device.
        </p>
      </section>

      <section className="card">
        <h2>1) Calibration Inputs</h2>
        <p className="sub">Add at least 3 matched Ideal and Measured points. Scroll to add more rows as needed.</p>
        <div className="grid head"><b>Ideal X</b><b>Ideal Y</b><b>Measured X</b><b>Measured Y</b></div>
        {pairs.map((p, i) => (
          <div className="grid row" key={i}>
            {(["ix", "iy", "mx", "my"] as const).map((k) => (
              <input key={k} type="number" value={p[k]} onChange={(e) => {
                const v = Number(e.target.value);
                setPairs((old) => old.map((x, j) => j === i ? { ...x, [k]: v } : x));
              }} />
            ))}
          </div>
        ))}
        <div className="actions">
          <button onClick={() => setPairs((p) => [...p, { ix: 0, iy: 0, mx: 0, my: 0 }])}>Add pair</button>
          <button onClick={onSolve}>Solve calibration</button>
        </div>
      </section>

      <section className="card">
        <h2>2) DXF Upload and Output</h2>
        <p className="sub">Supported parse/write focus: LINE, LWPOLYLINE, POLYLINE. Start with these for reliable round-trip.</p>
        <input type="file" accept=".dxf" onChange={(e) => e.target.files?.[0] && onUploadDxf(e.target.files[0])} />
        <div className="actions">
          <label><input type="checkbox" checked={showInput} onChange={(e) => setShowInput(e.target.checked)} /> Input</label>
          <label><input type="checkbox" checked={showOutput} onChange={(e) => setShowOutput(e.target.checked)} /> Output</label>
          <label><input type="checkbox" checked={showDistorted} onChange={(e) => setShowDistorted(e.target.checked)} /> Distorted</label>
          <label><input type="checkbox" checked={showCoords} onChange={(e) => setShowCoords(e.target.checked)} /> Coordinates</label>
          <button disabled={!affine || outputPaths.length === 0} onClick={onDownloadOutput}>Download compensated DXF</button>
        </div>
      </section>

      <section className="card">
        <h2>3) Interaction Tools</h2>
        <p className="sub">Use Measure or Reference mode to click points directly in the viewer. BBox corners are selectable when BBox is ON.</p>
        <div className="actions">
          <button
            onClick={() => {
              setMeasureMode((v) => !v);
              setReferenceMode(false);
              setMeasurePending(null);
            }}
          >
            {measureMode ? "Stop Measure" : "Measure DX/DY"}
          </button>
          <button onClick={() => { setMeasureRecords([]); setMeasurePending(null); }}>Clear measurements</button>
        </div>
        <div className="actions">
          <input type="number" value={refX} onChange={(e) => setRefX(e.target.value)} placeholder="Reference X" />
          <input type="number" value={refY} onChange={(e) => setRefY(e.target.value)} placeholder="Reference Y" />
          <button onClick={onSetReference}>Set reference</button>
          <button
            onClick={() => {
              if (!referencePoint) return;
              setReferenceMode((v) => !v);
              setMeasureMode(false);
              setMeasurePending(null);
            }}
            disabled={!referencePoint}
          >
            {referenceMode ? "Stop reference picks" : "Add dimensions"}
          </button>
          <button onClick={() => setReferenceRecords([])}>Clear dimensions</button>
          <button
            onClick={() => {
              setReferencePoint(null);
              setReferenceMode(false);
              setReferenceRecords([]);
            }}
          >
            Clear reference
          </button>
        </div>
        <div className="actions">
          <label><input type="checkbox" checked={bboxEnabled} onChange={(e) => setBboxEnabled(e.target.checked)} /> Bounding box</label>
          <label>
            Layer
            <select value={bboxLayer} onChange={(e) => setBboxLayer(e.target.value as LayerName)} style={{ marginLeft: 6 }}>
              <option>Input</option>
              <option>Output</option>
              <option>Distorted</option>
            </select>
          </label>
        </div>
        <div className="split">
          <div>
            <h3>Measurement log</h3>
            <div className="log">
              {measureRecords.length === 0 ? "No measurements yet." : measureRecords.map((r, i) => (
                <div key={`m-${i}`}>#{i + 1} {r.a.label} → {r.b.label} | DX={r.dx.toFixed(3)}, DY={r.dy.toFixed(3)}</div>
              ))}
            </div>
          </div>
          <div>
            <h3>Reference dimensions log</h3>
            <div className="log">
              {referenceRecords.length === 0 ? "No reference dimensions yet." : referenceRecords.map((r, i) => (
                <div key={`r-${i}`}>#{i + 1} {r.label} | DX={r.dx.toFixed(3)}, DY={r.dy.toFixed(3)}</div>
              ))}
            </div>
          </div>
        </div>
      </section>

      <section className="card plot">
        <h2>4) Compare Viewer</h2>
        <p className="sub">Pan/zoom and inspect layer behavior. Output uses inverse affine compensation. Distorted simulates machine execution from input.</p>
        <Plot
          data={plotData}
          layout={plotLayout}
          style={{ width: "100%" }}
          config={plotConfig}
          onClick={onPlotClick}
        />
      </section>

      {affine && (
        <section className="card">
          <h2>Affine Result</h2>
          <pre>{JSON.stringify(affine, null, 2)}</pre>
        </section>
      )}
      {error && <section className="card error">{error}</section>}
    </div>
  );
}
