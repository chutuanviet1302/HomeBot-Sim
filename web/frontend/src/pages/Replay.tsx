import { useEffect, useMemo, useRef, useState } from "react";
import { api, fmt, media, personVi, safetyVi, stateVi, type Replay, type ReplayInfo } from "../api";
import { useData } from "../charts";

const STATE_COLORS: Record<string, string> = {
  "NAVIGATE_TO(table)": "#60a5fa",
  "MANIPULATE(pick_cup)": "#a78bfa",
  "NAVIGATE_TO(sink)": "#34d399",
  "MANIPULATE(place_cup)": "#f59e0b",
  DONE: "#9ca3af",
};
const SPEEDS = [1, 2, 4, 8, 16];
const PLACE_VI: Record<string, string> = { living_room: "phòng khách", kitchen: "bếp", table: "bàn", sink: "bồn rửa", toy_box: "hộp đồ chơi" };

export default function ReplayPage({ replayKey }: { replayKey?: string }) {
  const list = useData(api.replays);
  const items = list.data ?? [];
  const current: ReplayInfo | undefined =
    items.find((r) => r.key === replayKey) ?? items.find((r) => r.key === "l2_s2") ?? items[0];
  const rep = useData(() => (current ? api.replay(current) : Promise.resolve(null)), [current?.key]);
  const missing = replayKey && list.data && !items.some((r) => r.key === replayKey);

  return (
    <section>
      <div className="section-head">
        <h1 style={{ marginBottom: 0 }}>Xem lại tập mô phỏng</h1>
        {items.length > 0 && (
          <label className="field" style={{ flexDirection: "row", alignItems: "center", gap: 8 }}>
            Tập
            <select value={current?.key} onChange={(e) => (window.location.hash = `#/replay/${e.target.value}`)}>
              {items.map((r) => (
                <option key={r.key} value={r.key}>
                  Mức {r.level} · seed {r.seed}
                  {r.person_profile === "elderly" ? " · người cao tuổi" : ""}
                  {r.sink_offset ? ` · bồn ${r.sink_offset > 0 ? "+" : ""}${r.sink_offset.toFixed(2)} m` : ""}
                  {r.policy !== "expert" ? ` · ${r.policy}` : ""} — {r.result === "success" ? "thành công" : r.result}
                </option>
              ))}
            </select>
          </label>
        )}
      </div>
      {list.error && <p className="error">Không tải được danh sách: {list.error}</p>}
      {missing && <p className="error">Chưa có bản ghi “{replayKey}”. Đang hiển thị tập khác.</p>}
      {list.data && !items.length && (
        <div className="empty">Chưa có bản ghi nào trong logs/. Hãy chạy một tập ở trang “Chạy mô phỏng” hoặc scripts/view.py.</div>
      )}
      {rep.error && <p className="error">Không tải được bản ghi: {rep.error}</p>}
      {rep.data && <Player key={rep.data.key} r={rep.data} />}
    </section>
  );
}

function Player({ r }: { r: Replay }) {
  const [t, setT] = useState(0); // seconds
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(4);
  const dur = (r.n - 1) / r.hz;
  const i = Math.min(r.n - 1, Math.max(0, Math.round(t * r.hz)));

  useEffect(() => {
    if (!playing) return;
    let raf = 0;
    let last = performance.now();
    const tick = (now: number) => {
      const dt = (now - last) / 1000;
      last = now;
      setT((x) => {
        const nx = x + dt * speed;
        if (nx >= dur) {
          setPlaying(false);
          return dur;
        }
        return nx;
      });
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [playing, speed, dur]);

  const segments = useMemo(() => {
    const out: { label: string; start: number; end: number }[] = [];
    r.state.forEach((s, k) => {
      const label = r.state_labels[s];
      const last = out[out.length - 1];
      if (last && last.label === label) last.end = k;
      else out.push({ label, start: k, end: k });
    });
    return out;
  }, [r]);

  const stats = useMemo(() => {
    let minD = Infinity;
    if (r.person) {
      for (let k = 0; k < r.n; k++) {
        const d = Math.hypot(r.robot.x[k] - r.person.x[k], r.robot.y[k] - r.person.y[k]);
        minD = Math.min(minD, d);
      }
    }
    const frac = (name: string) => {
      const idx = r.safety_labels.indexOf(name);
      return idx < 0 ? 0 : r.safety.filter((s) => s === idx).length / r.n;
    };
    let path = 0;
    for (let k = 1; k < r.n; k++) path += Math.hypot(r.robot.x[k] - r.robot.x[k - 1], r.robot.y[k] - r.robot.y[k - 1]);
    return { minD, stop: frac("stop"), slow: frac("slow"), path };
  }, [r]);

  const state = r.state_labels[r.state[i]];
  const safety = r.safety_labels[r.safety[i]];
  const atEnd = i === r.n - 1;

  return (
    <div className="replay-layout">
      <div className="panel" style={{ padding: 12 }}>
        <div className="canvas-wrap">
          <FloorCanvas r={r} i={i} />
          <div className="overlay">
            <span>{atEnd ? (r.result === "success" ? "Hoàn thành ✓" : r.result) : stateVi(state)}</span>
            {safety && <span style={{ color: safety === "stop" ? "var(--bad)" : safety === "clear" ? "var(--good)" : "var(--pet)" }}>an toàn: {safetyVi(safety)}</span>}
          </div>
        </div>
        <div className="controls">
          <button className="primary" onClick={() => (atEnd ? (setT(0), setPlaying(true)) : setPlaying(!playing))} aria-label="play">
            {playing ? "❚❚ Dừng" : atEnd ? "↺ Phát lại" : "▶ Phát"}
          </button>
          <input
            type="range"
            min={0}
            max={dur}
            step={1 / r.hz}
            value={t}
            onChange={(e) => {
              setT(Number(e.target.value));
            }}
            aria-label="thời gian"
          />
          <span className="num small" style={{ minWidth: 92, textAlign: "right" }}>
            {t.toFixed(1)} / {dur.toFixed(0)} s
          </span>
          <select value={speed} onChange={(e) => setSpeed(Number(e.target.value))} aria-label="tốc độ">
            {SPEEDS.map((s) => (
              <option key={s} value={s}>
                ×{s}
              </option>
            ))}
          </select>
        </div>
        <div className="timeline" title="Trạng thái nhiệm vụ theo thời gian">
          {segments.map((s, k) => (
            <div
              key={k}
              title={stateVi(s.label)}
              onClick={() => setT(s.start / r.hz)}
              style={{ flex: s.end - s.start + 1, background: STATE_COLORS[s.label] ?? "#9ca3af", cursor: "pointer" }}
            />
          ))}
        </div>
        <div className="legend">
          {[...new Set(segments.map((s) => s.label))].map((l) => (
            <span key={l}>
              <i style={{ background: STATE_COLORS[l] ?? "#9ca3af" }} />
              {stateVi(l)}
            </span>
          ))}
        </div>
      </div>
      <aside className="grid" style={{ gap: 16 }}>
        <div className="panel">
          <h3>Thông tin tập</h3>
          <div className="stat-list">
            <div><span>Lệnh</span>“{r.command}”</div>
            <div><span>Kết quả</span><b style={{ color: r.result === "success" ? "var(--good)" : "var(--bad)" }}>{r.result === "success" ? "thành công" : r.result}</b></div>
            <div><span>Mức · seed</span>{r.level} · {r.seed}</div>
            <div><span>Người trong nhà</span>{r.level >= 2 ? personVi(r.person_profile) : "không"}</div>
            <div><span>Thú cưng</span>{r.pet ? "có" : "không"}</div>
            <div><span>Bồn rửa dời</span>{fmt(r.sink_offset, 3, " m")}</div>
            <div><span>Chính sách</span>{r.policy}</div>
            <div><span>Thời lượng</span>{fmt(r.duration_s, 1, " s")}</div>
            <div><span>Quãng đường robot</span>{fmt(stats.path, 1, " m")}</div>
            {r.person && <div><span>K/c tâm gần nhất tới người</span>{fmt(stats.minD, 2, " m")}</div>}
            <div><span>Thời gian dừng / giảm tốc</span>{(100 * stats.stop).toFixed(0)}% / {(100 * stats.slow).toFixed(0)}%</div>
          </div>
        </div>
        <div className="panel">
          <h3>Chú thích</h3>
          <div className="legend" style={{ flexDirection: "column", gap: 6, marginTop: 0 }}>
            <span><i style={{ background: "var(--robot)", borderRadius: 99 }} />Robot G1 (vạch: hướng nhìn)</span>
            <span><i style={{ background: "var(--person)", borderRadius: 99 }} />Người</span>
            {r.pet && <span><i style={{ background: "var(--pet)" }} />Thú cưng</span>}
            <span><i style={{ background: "var(--can)", borderRadius: 99 }} />Lon</span>
            <span><i style={{ background: "var(--furniture-move)" }} />Đồ đạc bị xê dịch ngẫu nhiên</span>
            <span><i style={{ background: "transparent", border: "2px solid var(--accent)" }} />Bồn rửa</span>
          </div>
        </div>
        {r.key === "l2_s2" && <Video />}
      </aside>
    </div>
  );
}

function Video() {
  const [ok, setOk] = useState(true);
  const files = useData(api.media);
  if (!ok || !files.data?.includes("demo_full_task.mp4")) return null;
  return (
    <figure className="demo panel" style={{ margin: 0, padding: 0 }}>
      <video src={media("demo_full_task.mp4")} controls muted playsInline preload="metadata" onError={() => setOk(false)} />
      <figcaption>Video MuJoCo của cùng tập (mức 2, seed 2), có hiển thị đầu ra nhận thức.</figcaption>
    </figure>
  );
}

function cssVar(name: string) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || "#888";
}

function FloorCanvas({ r, i }: { r: Replay; i: number }) {
  const ref = useRef<HTMLCanvasElement>(null);
  const [w, setW] = useState(800);
  const [dark, setDark] = useState(() => window.matchMedia?.("(prefers-color-scheme: dark)").matches ?? false);

  useEffect(() => {
    const el = ref.current?.parentElement;
    if (!el) return;
    const ro = new ResizeObserver(() => setW(el.clientWidth));
    ro.observe(el);
    setW(el.clientWidth);
    const mq = window.matchMedia?.("(prefers-color-scheme: dark)");
    const on = () => setDark(mq.matches);
    mq?.addEventListener("change", on);
    return () => {
      ro.disconnect();
      mq?.removeEventListener("change", on);
    };
  }, []);

  useEffect(() => {
    const cv = ref.current;
    if (!cv || w <= 0) return;
    const fp = r.floorplan;
    const m = 0.15;
    const s = w / (fp.width + 2 * m);
    const h = (fp.height + 2 * m) * s;
    const dpr = window.devicePixelRatio || 1;
    cv.width = Math.round(w * dpr);
    cv.height = Math.round(h * dpr);
    cv.style.height = `${h}px`;
    const c = cv.getContext("2d")!;
    c.setTransform(dpr, 0, 0, dpr, 0, 0);
    const X = (x: number) => (x + m) * s;
    const Y = (y: number) => (fp.height + m - y) * s;
    const col = {
      floor: cssVar("--floor"), wall: cssVar("--wall"), furn: cssVar("--furniture"), move: cssVar("--furniture-move"),
      text: cssVar("--muted"), robot: cssVar("--robot"), person: cssVar("--person"), pet: cssVar("--pet"),
      can: cssVar("--can"), accent: cssVar("--accent"), border: cssVar("--border"),
    };
    c.clearRect(0, 0, w, h);
    c.fillStyle = col.floor;
    c.fillRect(X(0), Y(fp.height), fp.width * s, fp.height * s);

    // furniture and walls
    for (const it of fp.items) {
      c.save();
      c.translate(X(it.x), Y(it.y));
      c.fillStyle = it.wall ? col.wall : it.movable ? col.move : col.furn;
      c.strokeStyle = col.border;
      if (it.kind === "box") {
        c.rotate(-(it.yaw ?? 0));
        c.fillRect(-it.hx! * s, -it.hy! * s, 2 * it.hx! * s, 2 * it.hy! * s);
      } else {
        c.beginPath();
        c.arc(0, 0, it.r! * s, 0, 2 * Math.PI);
        c.fill();
      }
      c.restore();
    }
    // sink
    c.strokeStyle = col.accent;
    c.lineWidth = 2;
    c.strokeRect(X(fp.sink.x - fp.sink.hx), Y(fp.sink.y + fp.sink.hy), 2 * fp.sink.hx * s, 2 * fp.sink.hy * s);
    // place labels
    c.fillStyle = col.text;
    c.font = `${Math.max(10, Math.min(13, s * 0.17))}px system-ui, sans-serif`;
    c.textAlign = "center";
    for (const [k, p] of Object.entries(fp.places)) {
      if (k === "sink") continue;
      c.fillText(PLACE_VI[k] ?? k, X(p.x), Y(p.y) + 4);
    }
    c.fillText(PLACE_VI.sink, X(fp.sink.x), Y(fp.sink.y + fp.sink.hy) - 6);

    const trail = (xs: number[], ys: number[], color: string, width: number, from = 0) => {
      c.strokeStyle = color;
      c.lineWidth = width;
      c.globalAlpha = 0.6;
      c.lineJoin = "round";
      c.beginPath();
      for (let k = from; k <= i; k++) {
        if (k === from) c.moveTo(X(xs[k]), Y(ys[k]));
        else c.lineTo(X(xs[k]), Y(ys[k]));
      }
      c.stroke();
      c.globalAlpha = 1;
    };
    const tail = Math.max(0, i - 20 * r.hz); // people / pet: last 20 s
    if (r.person) trail(r.person.x, r.person.y, col.person, 1.5, tail);
    if (r.pet) trail(r.pet.x, r.pet.y, col.pet, 1.5, tail);
    trail(r.robot.x, r.robot.y, col.robot, 2.5);
    trail(r.can.x, r.can.y, col.can, 2);

    // bodies
    const disc = (x: number, y: number, rad: number, color: string, alpha = 0.85) => {
      c.globalAlpha = alpha;
      c.fillStyle = color;
      c.beginPath();
      c.arc(X(x), Y(y), Math.max(3, rad * s), 0, 2 * Math.PI);
      c.fill();
      c.globalAlpha = 1;
    };
    if (r.person) disc(r.person.x[i], r.person.y[i], fp.person_radius, col.person);
    if (r.pet) {
      c.save();
      c.translate(X(r.pet.x[i]), Y(r.pet.y[i]));
      c.rotate(-r.pet.yaw[i]);
      c.fillStyle = col.pet;
      c.fillRect(-fp.pet_half[0] * s, -fp.pet_half[1] * s, 2 * fp.pet_half[0] * s, 2 * fp.pet_half[1] * s);
      c.restore();
    }
    const rx = r.robot.x[i], ry = r.robot.y[i], yaw = r.robot.yaw[i];
    disc(rx, ry, fp.robot_radius, col.robot, 0.25);
    disc(rx, ry, 0.13, col.robot, 1);
    c.strokeStyle = col.robot;
    c.lineWidth = 3;
    c.beginPath();
    c.moveTo(X(rx), Y(ry));
    c.lineTo(X(rx + 0.4 * Math.cos(yaw)), Y(ry + 0.4 * Math.sin(yaw)));
    c.stroke();
    disc(r.can.x[i], r.can.y[i], 0.04, col.can, 1);
  }, [r, i, w, dark]);

  return <canvas ref={ref} aria-label="Sơ đồ căn hộ nhìn từ trên xuống" />;
}
