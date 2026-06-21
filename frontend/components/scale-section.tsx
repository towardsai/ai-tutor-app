import { SCALES } from "@/lib/experiments";

export function ScaleSection() {
  return (
    <div className="scale-grid">
      {SCALES.map((strip) => {
        const max = Math.max(...strip.items.map((i) => i.value));
        return (
          <div key={strip.title} className="scale-card">
            <h3 className="scale-title">
              {strip.title}
              <span className="scale-unit">{strip.unit}</span>
            </h3>
            <div className="scale-items">
              {strip.items.map((item) => (
                <div key={item.label} className="scale-item">
                  <div className="scale-item-head">
                    <span className="scale-item-label">{item.label}</span>
                    <span className="scale-item-value">{item.display}</span>
                  </div>
                  <div className="scale-track">
                    <div
                      className="scale-fill"
                      style={{ width: `${Math.max(4, (item.value / max) * 100)}%` }}
                    />
                  </div>
                  <span className="scale-note">{item.note}</span>
                </div>
              ))}
            </div>
          </div>
        );
      })}
    </div>
  );
}
