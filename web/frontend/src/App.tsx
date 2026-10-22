import { useEffect, useState } from "react";
import { STATIC } from "./api";
import Overview from "./pages/Overview";
import Experiments from "./pages/Experiments";
import ReplayPage from "./pages/Replay";
import RunSim from "./pages/RunSim";

// Tiny hash router: #/, #/experiments, #/replay/<key>, #/run
function useRoute(): string[] {
  const parse = () => window.location.hash.replace(/^#\/?/, "").split("/").filter(Boolean);
  const [route, setRoute] = useState(parse);
  useEffect(() => {
    const on = () => {
      setRoute(parse());
      window.scrollTo(0, 0);
    };
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);
  return route;
}

const NAV = [
  { id: "", label: "Tổng quan" },
  { id: "experiments", label: "Thí nghiệm" },
  { id: "replay", label: "Xem lại" },
  ...(STATIC ? [] : [{ id: "run", label: "Chạy mô phỏng" }]),
];

export default function App() {
  const route = useRoute();
  const page = route[0] ?? "";
  return (
    <>
      <header className="topbar">
        <div className="topbar-inner">
          <a className="brand" href="#/">
            <span className="brand-mark" aria-hidden>
              <svg viewBox="0 0 32 32" width="22" height="22">
                <rect x="4" y="4" width="24" height="24" rx="7" fill="currentColor" />
                <circle cx="12" cy="14" r="2.5" fill="var(--surface)" />
                <circle cx="20" cy="14" r="2.5" fill="var(--surface)" />
                <rect x="11" y="20" width="10" height="2.5" rx="1.2" fill="var(--surface)" />
              </svg>
            </span>
            HomeBot-Sim
          </a>
          <nav className="nav">
            {NAV.map((n) => (
              <a key={n.id} href={`#/${n.id}`} className={page === n.id ? "active" : ""}>
                {n.label}
              </a>
            ))}
          </nav>
        </div>
      </header>
      <main className="container">
        {page === "" && <Overview />}
        {page === "experiments" && <Experiments />}
        {page === "replay" && <ReplayPage replayKey={route[1]} />}
        {page === "run" && (STATIC ? <p className="empty">Bản tĩnh chỉ hiển thị kết quả; hãy chạy máy chủ (web/README.md) để mô phỏng.</p> : <RunSim />)}
        {!["", "experiments", "replay", "run"].includes(page) && <p className="muted">Không tìm thấy trang.</p>}
      </main>
      <footer className="footer">
        HomeBot-Sim · Unitree G1 + bàn tay Inspire trong MuJoCo · {STATIC ? "bản tĩnh" : "dữ liệu trực tiếp từ results/"}
      </footer>
    </>
  );
}
