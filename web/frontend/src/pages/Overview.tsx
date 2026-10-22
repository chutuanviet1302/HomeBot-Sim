import { useState } from "react";
import { api, fmt, media, pct, type Card, type Kpi } from "../api";
import { DotStrip, GroupedBars, HBars, useData } from "../charts";

const POLICY_COLORS = ["var(--accent)", "#8b5cf6", "var(--pet)", "var(--can)"];
const EXP_SHORT: Record<string, string> = {
  "E2-pick": "Nhặt lon",
  "E2-place": "Đặt vào bồn",
  "E2-place-sink-moved": "Đặt, bồn dời",
  E3: "Toàn nhiệm vụ",
  "E3-sink-moved": "Toàn bộ, bồn dời",
  E5: "Người cao tuổi",
};
const LEVEL_NAME: Record<string, string> = {
  "E1-L1": "Mức 1 · nhà trống",
  "E1-L2": "Mức 2 · có người",
  "E1-L3": "Mức 3 · người + thú",
  E4: "Mức 2 · tắt an toàn",
};

export default function Overview() {
  const { data, error } = useData(api.kpi);
  return (
    <>
      <Hero />
      {error && <p className="error">Không tải được dữ liệu: {error}</p>}
      {data && (
        <>
          <section>
            <div className="section-head">
              <h2>Kết quả chính</h2>
              <p className="muted small">Tự động tính từ các file CSV trong results/</p>
            </div>
            <div className="cards">
              {data.cards.map((c) => (
                <KpiCard key={c.id} c={c} />
              ))}
            </div>
          </section>
          <Safety k={data} />
          <Levels k={data} />
          <Policies k={data} />
          <Elderly k={data} />
        </>
      )}
    </>
  );
}

function Hero() {
  const [gifOk, setGifOk] = useState(true);
  const files = useData(api.media);
  return (
    <section className="hero">
      <div>
        <div className="eyebrow">Robot hỗ trợ người cao tuổi · mô phỏng</div>
        <h1>Dọn dẹp giúp người già và người khuyết tật — an toàn khi có người trong nhà</h1>
        <p className="lead">
          Người cao tuổi hoặc người đi lại khó khăn thường phải tự mang cốc, lon đã dùng ra bồn rửa — một việc nhỏ
          nhưng dễ gây té ngã. HomeBot-Sim đặt robot hình người Unitree G1 với bàn tay Inspire vào một căn hộ mô phỏng
          (MuJoCo) và giao lệnh <b>“mang lon tới bồn rửa”</b>, trong khi đồ đạc bị xê dịch và một người đi lại quanh
          nhà.
        </p>
        <div className="needs">
          <div className="need">
            <div className="need-icon">1</div>
            <div>
              <b>Phải làm được việc</b>
              <span>Tìm lon trên bàn, nhặt lên, mang qua cửa, tìm bồn bằng camera và đặt vào.</span>
            </div>
          </div>
          <div className="need">
            <div className="need-icon">2</div>
            <div>
              <b>An toàn với con người</b>
              <span>Giảm tốc, dừng và né khi người đến gần; cánh tay dừng khi có va chạm; không được ngã.</span>
            </div>
          </div>
          <div className="need">
            <div className="need-icon">3</div>
            <div>
              <b>Không cần lập trình riêng cho từng nhà</b>
              <span>Đồ đạc xê dịch, bồn rửa dời chỗ: robot tự nhận thức bằng LiDAR và camera.</span>
            </div>
          </div>
        </div>
      </div>
      {!files.data ? (
        <div />
      ) : gifOk && files.data.includes("demo.gif") ? (
        <figure className="demo" style={{ margin: 0 }}>
          <img src={media("demo.gif")} alt="Robot G1 mang lon tới bồn rửa trong khi một người đi lại" onError={() => setGifOk(false)} />
          <figcaption>Mức 2, seed 2: một người (trụ đỏ) đi lại, tua nhanh 7 lần.</figcaption>
        </figure>
      ) : (
        <div className="empty">Chưa có ảnh động demo (report/demo.gif).</div>
      )}
    </section>
  );
}

function KpiCard({ c }: { c: Card }) {
  return (
    <div className={`card ${c.tone ?? ""}`}>
      <div className="card-label">{c.label}</div>
      <div className="card-value">
        {c.value}
        {c.compare && (
          <span className="vs">
            vs <b>{c.compare}</b>
          </span>
        )}
      </div>
      {c.pct !== undefined && c.pct !== null && (
        <div className="meter" aria-label={`${c.pct}%`}>
          <div style={{ width: `${c.pct}%` }} />
        </div>
      )}
      {c.sub && <div className="card-sub">{c.sub}</div>}
    </div>
  );
}

function Safety({ k }: { k: Kpi }) {
  const s = k.safety;
  if (!s) return null;
  const a = s.with;
  const b = s.without;
  return (
    <section>
      <div className="section-head">
        <h2>Lớp an toàn có giá trị gì? (E4)</h2>
        <p className="muted small">
          Cùng {a.n} cảnh mức 2 (có người đi lại), bật vs. tắt lớp an toàn
        </p>
      </div>
      <div className="grid grid-2">
        <div className="panel">
          <h3>Va chạm với người khi robot đang di chuyển</h3>
          <HBars
            bars={[
              { label: "Bật an toàn", value: a.person_contacts_robot_moving, color: "var(--good)" },
              { label: "Tắt an toàn", value: b.person_contacts_robot_moving, color: "var(--bad)" },
            ]}
          />
          <h3 style={{ marginTop: 14 }}>Số lượt có ít nhất một va chạm</h3>
          <HBars
            max={a.n}
            bars={[
              { label: "Bật an toàn", value: a.runs_with_contact, text: `${a.runs_with_contact}/${a.n}`, color: "var(--good)" },
              { label: "Tắt an toàn", value: b.runs_with_contact, text: `${b.runs_with_contact}/${b.n}`, color: "var(--bad)" },
            ]}
          />
          <p className="muted small" style={{ marginTop: 8 }}>
            Tỉ lệ đến bàn: {pct(a.success_rate)} (bật) vs. {pct(b.success_rate)} (tắt) — an toàn hơn mà gần như không mất
            hiệu quả.
          </p>
        </div>
        <div className="panel">
          <h3>Khoảng cách gần nhất tới người trong mỗi lượt</h3>
          <DotStrip
            clampMax={2.5}
            groups={[
              { label: "Bật an toàn", values: s.min_dist_with, color: "var(--good)" },
              { label: "Tắt an toàn", values: s.min_dist_without, color: "var(--bad)" },
            ]}
          />
          <p className="muted small" style={{ marginTop: 8 }}>
            Mỗi chấm là một lượt; vạch đậm là trung vị ({fmt(a.min_person_dist_median)} m vs. {fmt(b.min_person_dist_median)} m).
            Giá trị âm (bên trái đường đứt) nghĩa là thân robot chạm vào người. Thấp nhất: {fmt(a.min_person_dist_min)} m vs.{" "}
            {fmt(b.min_person_dist_min)} m.
          </p>
        </div>
      </div>
    </section>
  );
}

function Levels({ k }: { k: Kpi }) {
  if (!k.levels.length) return null;
  return (
    <section>
      <div className="section-head">
        <h2>Điều hướng theo độ khó (E1)</h2>
        <p className="muted small">Đi từ phòng khách tới bàn ăn; đồ đạc xê dịch mỗi lượt</p>
      </div>
      <div className="panel">
        <HBars
          max={1}
          bars={k.levels.map((l) => ({
            label: LEVEL_NAME[l.experiment] ?? l.experiment,
            value: l.success_rate ?? 0,
            text: `${pct(l.success_rate)} · ${fmt(l.time_s_mean_success, 0)} s`,
            color: l.experiment === "E4" ? "var(--bad)" : "var(--accent)",
          }))}
        />
        <p className="muted small" style={{ marginTop: 8 }}>
          Tỉ lệ thành công · thời gian trung bình của lượt thành công. Robot ngã: {k.levels.reduce((s, l) => s + l.falls, 0)} lần.
        </p>
      </div>
    </section>
  );
}

function Policies({ k }: { k: Kpi }) {
  const cats = [...new Set(k.policies.map((p) => p.experiment))];
  const pols = [...new Set(k.policies.map((p) => p.policy))].sort((a, b) =>
    a === "expert" ? -1 : b === "expert" ? 1 : a.localeCompare(b),
  );
  return (
    <section>
      <div className="section-head">
        <h2>Chuyên gia vs. chính sách học (ACT)</h2>
        <p className="muted small">Tỉ lệ thành công theo kỹ năng / nhiệm vụ</p>
      </div>
      <div className="panel">
        {cats.length ? (
          <>
            <GroupedBars
              min={cats.length * 90}
              categories={cats.map((c) => EXP_SHORT[c] ?? c)}
              series={pols.map((p, i) => {
                const rows = cats.map((c) => k.policies.find((x) => x.experiment === c && x.policy === p));
                return {
                  name: p,
                  color: POLICY_COLORS[i % POLICY_COLORS.length],
                  values: rows.map((r) => (r ? r.success_rate : null)),
                  labels: rows.map((r) => (r ? `${r.success}/${r.n}` : "")),
                };
              })}
            />
            <div className="legend">
              {pols.map((p, i) => (
                <span key={p}>
                  <i style={{ background: POLICY_COLORS[i % POLICY_COLORS.length] }} />
                  {p === "expert" ? "Chuyên gia (lập kế hoạch + nhận thức)" : p}
                </span>
              ))}
            </div>
            {!k.has_learned && (
              <p className="muted small" style={{ marginTop: 10 }}>
                Chưa có kết quả của chính sách học (ACT); biểu đồ sẽ tự cập nhật khi có file results/e2_manip_act*.csv
                hoặc results/e3_full_level2_act*.csv.
              </p>
            )}
          </>
        ) : (
          <div className="empty">Chưa có kết quả thao tác.</div>
        )}
      </div>
    </section>
  );
}

function Elderly({ k }: { k: Kpi }) {
  return (
    <section>
      <div className="section-head">
        <h2>Sống cùng người cao tuổi (E5)</h2>
        <p className="muted small">Người đi chậm 0.4–0.7 m/s, hay dừng và do dự</p>
      </div>
      {k.e5 ? (
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Nhóm</th>
                <th className="r">Lượt</th>
                <th className="r">Thành công</th>
                <th className="r">Va chạm (đang đi)</th>
                <th className="r">K/c tối thiểu (trung vị)</th>
                <th className="r">Thời gian TB</th>
                <th className="r">Ngã</th>
              </tr>
            </thead>
            <tbody>
              {[
                ...k.e5.groups,
                ...(k.e5.reference_e3 ? [{ label: "Tham chiếu E3 · người trưởng thành", ...k.e5.reference_e3 }] : []),
              ].map((g) => (
                <tr key={g.label} style={{ cursor: "default" }}>
                  <td>{g.label}</td>
                  <td className="r">{g.n}</td>
                  <td className="r">
                    {g.success}/{g.n} ({pct(g.success_rate)})
                  </td>
                  <td className="r">{g.person_contacts_robot_moving}</td>
                  <td className="r">{fmt(g.min_person_dist_median, 2, " m")}</td>
                  <td className="r">{fmt(g.time_s_mean_success, 0, " s")}</td>
                  <td className="r">{g.falls}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <div className="empty">Chưa có kết quả E5 (results/e5_elderly.csv). Trang sẽ hiển thị ngay khi file được tạo và nạp lại.</div>
      )}
    </section>
  );
}
