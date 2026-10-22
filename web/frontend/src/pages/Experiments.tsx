import { Fragment, useMemo, useState } from "react";
import { api, fmt, stateVi, type Experiment, type Run } from "../api";
import { useData } from "../charts";

type Col = { key: keyof Run; label: string; num?: boolean; render?: (r: Run) => string };
const COLS: Col[] = [
  { key: "id", label: "#", num: true },
  { key: "experiment", label: "Thí nghiệm" },
  { key: "seed", label: "Seed", num: true },
  { key: "policy", label: "Chính sách" },
  { key: "success", label: "Kết quả" },
  { key: "failed_at", label: "Lỗi tại", render: (r) => (r.failed_at ? stateVi(r.failed_at) : "") },
  { key: "time_s", label: "Thời gian (s)", num: true, render: (r) => fmt(r.time_s, 1) },
  { key: "person_contacts_robot_moving", label: "Va chạm người", num: true },
  { key: "min_person_dist", label: "K/c min (m)", num: true, render: (r) => fmt(r.min_person_dist, 3) },
  { key: "furniture_contacts", label: "Chạm đồ đạc", num: true },
  { key: "arm_stops", label: "Tay dừng", num: true },
  { key: "fell", label: "Ngã", render: (r) => (r.fell === null ? "–" : r.fell ? "có" : "không") },
  { key: "basin_err_mm", label: "Sai số bồn (mm)", num: true, render: (r) => fmt(r.basin_err_mm, 1) },
  { key: "perception_err_mm", label: "Sai số lon (mm)", num: true, render: (r) => fmt(r.perception_err_mm, 1) },
  { key: "sink_offset", label: "Dời bồn (m)", num: true, render: (r) => fmt(r.sink_offset, 3) },
];

export default function Experiments() {
  const exps = useData(api.experiments);
  const runs = useData(api.runs);
  const [exp, setExp] = useState("");
  const [policy, setPolicy] = useState("");
  const [success, setSuccess] = useState("");
  const [sort, setSort] = useState<{ key: keyof Run; dir: 1 | -1 }>({ key: "id", dir: 1 });
  const [open, setOpen] = useState<number | null>(null);

  const policies = useMemo(() => [...new Set((runs.data ?? []).map((r) => r.policy))].sort(), [runs.data]);
  const rows = useMemo(() => {
    const out = (runs.data ?? []).filter(
      (r) =>
        (!exp || r.experiment === exp) &&
        (!policy || r.policy === policy) &&
        (!success || String(Boolean(r.success)) === success),
    );
    const k = sort.key;
    return out.sort((a, b) => {
      const x = a[k] as unknown as number | string | null;
      const y = b[k] as unknown as number | string | null;
      if (x === y) return a.id - b.id;
      if (x === null || x === undefined) return 1;
      if (y === null || y === undefined) return -1;
      return (x < y ? -1 : 1) * sort.dir;
    });
  }, [runs.data, exp, policy, success, sort]);

  const nOk = rows.filter((r) => r.success).length;
  const error = exps.error || runs.error;

  return (
    <>
      <section>
        <div className="section-head">
          <h1 style={{ marginBottom: 0 }}>Thí nghiệm</h1>
          <p className="muted small">Mỗi dòng là một lượt chạy mô phỏng; bấm vào dòng để xem chi tiết.</p>
        </div>
        {error && <p className="error">Không tải được dữ liệu: {error}</p>}
        {exps.data && (
          <div className="chips">
            <button className={`chip ${exp === "" ? "active" : ""}`} onClick={() => setExp("")}>
              Tất cả <span className="muted">{runs.data?.length ?? ""}</span>
            </button>
            {exps.data.map((e: Experiment) => (
              <button
                key={e.id}
                className={`chip ${exp === e.id ? "active" : ""}`}
                onClick={() => setExp(exp === e.id ? "" : e.id)}
                title={e.description}
              >
                {e.name} <span className="muted">{e.success}/{e.n}</span>
              </button>
            ))}
          </div>
        )}
        {exp && exps.data && (
          <p className="muted small">
            {exps.data.find((e) => e.id === exp)?.description} Nguồn: {exps.data.find((e) => e.id === exp)?.source}
          </p>
        )}
        <div className="toolbar">
          <label className="field">
            Chính sách
            <select value={policy} onChange={(e) => setPolicy(e.target.value)}>
              <option value="">Tất cả</option>
              {policies.map((p) => (
                <option key={p}>{p}</option>
              ))}
            </select>
          </label>
          <label className="field">
            Kết quả
            <select value={success} onChange={(e) => setSuccess(e.target.value)}>
              <option value="">Tất cả</option>
              <option value="true">Thành công</option>
              <option value="false">Thất bại</option>
            </select>
          </label>
          <div style={{ flex: 1 }} />
          <span className="muted small num">
            {rows.length} lượt · {nOk} thành công{rows.length ? ` (${((100 * nOk) / rows.length).toFixed(0)}%)` : ""}
          </span>
          <button onClick={() => exportCsv(rows)} disabled={!rows.length}>
            Xuất CSV
          </button>
        </div>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                {COLS.map((c) => (
                  <th
                    key={c.key}
                    className={c.num ? "r" : ""}
                    onClick={() => setSort({ key: c.key, dir: sort.key === c.key ? (-sort.dir as 1 | -1) : 1 })}
                  >
                    {c.label}
                    {sort.key === c.key ? (sort.dir === 1 ? " ▲" : " ▼") : ""}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <Fragment key={r.id}>
                  <tr onClick={() => setOpen(open === r.id ? null : r.id)}>
                    {COLS.map((c) => (
                      <td key={c.key} className={c.num ? "r" : ""}>
                        {c.key === "success" ? (
                          <span className={`badge ${r.success ? "ok" : "fail"}`}>{r.success ? "Thành công" : "Thất bại"}</span>
                        ) : c.render ? (
                          c.render(r)
                        ) : (
                          String(r[c.key] ?? "–")
                        )}
                      </td>
                    ))}
                  </tr>
                  {open === r.id && (
                    <tr className="detail">
                      <td colSpan={COLS.length}>
                        <RunDetail r={r} />
                      </td>
                    </tr>
                  )}
                </Fragment>
              ))}
            </tbody>
          </table>
          {runs.data && !rows.length && <div className="empty" style={{ border: 0 }}>Không có lượt nào khớp bộ lọc.</div>}
        </div>
      </section>
    </>
  );
}

type Step = { state: string; arg: string | null; ok: boolean; t: number };

function RunDetail({ r }: { r: Run }) {
  const { trace, ...rest } = r.extra as { trace?: Step[] } & Record<string, unknown>;
  const level = (rest.level as number | undefined) ?? (r.experiment.match(/L(\d)/)?.[1] ? Number(r.experiment.match(/L(\d)/)![1]) : undefined);
  return (
    <div>
      <div className="kv">
        {Object.entries(rest).map(([k, v]) => (
          <div key={k}>
            <span>{k}: </span>
            {v === null ? "–" : typeof v === "object" ? JSON.stringify(v) : String(v)}
          </div>
        ))}
      </div>
      {Array.isArray(trace) && (
        <div className="trace">
          {trace.map((s, i) => (
            <div key={i} className={s.ok ? "" : "bad"}>
              {stateVi(s.arg ? `${s.state}(${s.arg})` : s.state)} {s.t ? `· ${s.t.toFixed(0)} s` : ""} {s.ok ? "✓" : "✗"}
            </div>
          ))}
        </div>
      )}
      {level !== undefined && (r.experiment.startsWith("E3") || r.experiment === "E5") && (
        <p className="small" style={{ marginTop: 10, marginBottom: 0 }}>
          <a href={`#/replay/l${level}_s${r.seed}`}>Xem lại tập mức {level}, seed {r.seed}</a>
          <span className="muted"> (nếu đã ghi; có thể ghi ở trang Chạy mô phỏng)</span>
        </p>
      )}
    </div>
  );
}

function exportCsv(rows: Run[]) {
  const extraKeys = [...new Set(rows.flatMap((r) => Object.keys(r.extra)))];
  const base = COLS.map((c) => c.key as string);
  const head = [...base, ...extraKeys];
  const esc = (v: unknown) => {
    if (v === null || v === undefined) return "";
    const s = typeof v === "object" ? JSON.stringify(v) : String(v);
    return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };
  const lines = [head.join(",")];
  for (const r of rows) {
    lines.push([...base.map((k) => esc(r[k as keyof Run])), ...extraKeys.map((k) => esc(r.extra[k]))].join(","));
  }
  const blob = new Blob(["﻿" + lines.join("\n")], { type: "text/csv;charset=utf-8" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "homebot_runs.csv";
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}
