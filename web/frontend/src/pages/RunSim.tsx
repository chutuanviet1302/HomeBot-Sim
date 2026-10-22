import { useEffect, useState } from "react";
import { api, personVi, type Job } from "../api";
import { useData } from "../charts";

const STATUS_VI: Record<Job["status"], string> = {
  queued: "đang chờ",
  running: "đang mô phỏng",
  done: "xong",
  failed: "lỗi",
};

export default function RunSim() {
  const policies = useData(api.policies);
  const [level, setLevel] = useState(2);
  const [seed, setSeed] = useState(0);
  const [sinkRange, setSinkRange] = useState(0);
  const [person, setPerson] = useState("adult");
  const [policy, setPolicy] = useState("expert");
  const [job, setJob] = useState<Job | null>(null);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const refreshJobs = () => api.jobs().then(setJobs).catch(() => undefined);
  useEffect(() => {
    refreshJobs();
  }, []);

  // poll the active job
  useEffect(() => {
    if (!job || job.status === "done" || job.status === "failed") return;
    const id = setInterval(() => {
      api
        .job(job.id)
        .then((j) => {
          setJob(j);
          if (j.status === "done" || j.status === "failed") refreshJobs();
        })
        .catch((e) => setErr(String(e.message ?? e)));
    }, 1500);
    return () => clearInterval(id);
  }, [job?.id, job?.status]);

  const submit = async () => {
    setErr(null);
    setBusy(true);
    try {
      const j = await api.createJob({ level, seed, sink_range: sinkRange, person, policy });
      setJob(j);
      refreshJobs();
    } catch (e) {
      setErr(String((e as Error).message ?? e));
    } finally {
      setBusy(false);
    }
  };

  const active = job && (job.status === "queued" || job.status === "running");

  return (
    <section>
      <div className="section-head">
        <h1 style={{ marginBottom: 0 }}>Chạy mô phỏng</h1>
        <p className="muted small">Mô phỏng một tập “mang lon tới bồn rửa” trên máy chủ (2–4 phút), rồi xem lại.</p>
      </div>
      <div className="grid grid-2">
        <div className="panel">
          <div className="form-grid">
            <div className="field">
              Mức độ khó
              <div className="seg">
                {[1, 2, 3].map((l) => (
                  <button key={l} className={level === l ? "on" : ""} onClick={() => setLevel(l)}>
                    {l}
                  </button>
                ))}
              </div>
              <span className="small">{["", "Nhà trống", "Có một người", "Người + thú cưng"][level]}</span>
            </div>
            <label className="field">
              Seed (bố trí nhà, vị trí xuất phát)
              <input type="number" min={0} max={99999} value={seed} onChange={(e) => setSeed(Math.max(0, Number(e.target.value) || 0))} />
            </label>
            <label className="field">
              Dời bồn rửa tối đa: ±{sinkRange.toFixed(2)} m
              <input type="range" min={0} max={0.3} step={0.01} value={sinkRange} onChange={(e) => setSinkRange(Number(e.target.value))} />
            </label>
            <div className="field">
              Người trong nhà
              <div className="seg">
                {["adult", "elderly"].map((p) => (
                  <button key={p} className={person === p ? "on" : ""} onClick={() => setPerson(p)} disabled={level < 2}>
                    {personVi(p)}
                  </button>
                ))}
              </div>
            </div>
            <label className="field">
              Chính sách thao tác
              <select value={policy} onChange={(e) => setPolicy(e.target.value)}>
                {(policies.data ?? ["expert"]).map((p) => (
                  <option key={p} value={p}>
                    {p === "expert" ? "Chuyên gia (expert)" : p}
                  </option>
                ))}
              </select>
            </label>
          </div>
          <div style={{ marginTop: 16, display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap" }}>
            <button className="primary" onClick={submit} disabled={busy || !!active}>
              {active ? "Đang chạy…" : "Bắt đầu mô phỏng"}
            </button>
            <span className="muted small">Mỗi lần chỉ chạy một tập; yêu cầu mới sẽ xếp hàng.</span>
          </div>
          {err && <p className="error" style={{ marginTop: 12 }}>{err}</p>}
        </div>
        <div className="panel">
          <h3>Tiến trình</h3>
          {job ? <JobView job={job} /> : <p className="muted">Chưa có tập nào được gửi trong phiên này.</p>}
        </div>
      </div>
      {jobs.length > 0 && (
        <section style={{ marginTop: 24 }}>
          <h2>Các lần chạy gần đây</h2>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>#</th>
                  <th>Mức</th>
                  <th>Seed</th>
                  <th>Người</th>
                  <th>Chính sách</th>
                  <th>Trạng thái</th>
                  <th>Kết quả</th>
                  <th></th>
                </tr>
              </thead>
              <tbody>
                {jobs.map((j) => (
                  <tr key={j.id} onClick={() => setJob(j)}>
                    <td>{j.id}</td>
                    <td>{j.level}</td>
                    <td>{j.seed}</td>
                    <td>{personVi(j.person)}</td>
                    <td>{j.policy}</td>
                    <td>{STATUS_VI[j.status]}</td>
                    <td>{j.result ?? ""}</td>
                    <td>{j.replay_key && <a href={`#/replay/${j.replay_key}`} onClick={(e) => e.stopPropagation()}>Xem lại</a>}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}
    </section>
  );
}

function JobView({ job }: { job: Job }) {
  return (
    <div>
      <p className="small" style={{ marginBottom: 8 }}>
        Tập #{job.id}: mức {job.level}, seed {job.seed}, {personVi(job.person)}, {job.policy} —{" "}
        <b style={{ color: job.status === "failed" ? "var(--bad)" : job.status === "done" ? "var(--good)" : undefined }}>
          {STATUS_VI[job.status]}
        </b>
      </p>
      <div className="progress" role="progressbar" aria-valuenow={Math.round(job.progress * 100)} aria-valuemin={0} aria-valuemax={100}>
        <div style={{ width: `${Math.round(job.progress * 100)}%` }} />
      </div>
      <p className="muted small num" style={{ marginTop: 6 }}>
        {Math.round(job.progress * 100)}% · đã mô phỏng {job.sim_t.toFixed(0)} s
      </p>
      {job.status === "done" && job.replay_key && (
        <p>
          Kết quả: <b>{job.result}</b>.{" "}
          <a className="btn primary" href={`#/replay/${job.replay_key}`}>
            Xem lại tập này
          </a>
        </p>
      )}
      {job.status === "failed" && <p className="error">Mô phỏng lỗi: {job.result}</p>}
      {job.log_tail.length > 0 && <div className="log">{job.log_tail.join("\n")}</div>}
    </div>
  );
}
