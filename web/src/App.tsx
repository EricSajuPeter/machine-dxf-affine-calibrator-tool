import { useMemo, useState } from "react";
import Plot from "react-plotly.js";
import { applyAffine, invertAffine, solveAffine, type Affine, type Point } from "./affine";
import { parseDxfToPaths, transformPaths, writeSimpleDxfFromPaths, type Path2D } from "./dxf";

type Pair = { ix: number; iy: number; mx: number; my: number };

const defaultPairs: Pair[] = [
  { ix: 0, iy: 0, mx: 0, my: 0 },
  { ix: 100, iy: 0, mx: 102, my: 1 },
  { ix: 0, iy: 100, mx: -1, my: 98 }
];

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

  const outputPaths = useMemo(() => {
    if (!affine) return [];
    const comp = invertAffine(affine);
    return transformPaths(inputPaths, comp);
  }, [inputPaths, affine]);

  const distortedPaths = useMemo(() => (affine ? transformPaths(inputPaths, affine) : []), [inputPaths, affine]);

  const traces = useMemo(() => {
    const t: any[] = [];
    if (showInput) t.push(...pathsToTraces(inputPaths, "#2c7be5", "Input"));
    if (showOutput) t.push(...pathsToTraces(outputPaths, "#00a86b", "Output"));
    if (showDistorted) t.push(...pathsToTraces(distortedPaths, "#8e44ad", "Distorted"));
    return t;
  }, [inputPaths, outputPaths, distortedPaths, showInput, showOutput, showDistorted]);

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
          <button disabled={!affine || outputPaths.length === 0} onClick={onDownloadOutput}>Download compensated DXF</button>
        </div>
      </section>

      <section className="card plot">
        <h2>3) Compare Viewer</h2>
        <p className="sub">Pan/zoom and inspect layer behavior. Output uses inverse affine compensation. Distorted simulates machine execution from input.</p>
        <Plot
          data={traces as any}
          layout={{ autosize: true, height: 560, paper_bgcolor: "white", plot_bgcolor: "white", xaxis: { title: "X" }, yaxis: { title: "Y", scaleanchor: "x" } }}
          style={{ width: "100%" }}
          config={{ responsive: true }}
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
