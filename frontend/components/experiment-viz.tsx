"use client";

import { useEffect, useState } from "react";
import clsx from "clsx";
import type { Bar, ResultGroup, View } from "@/lib/experiments";

function BarRow({ b, accent, shown }: { b: Bar; accent: string; shown: boolean }) {
  const color = b.tone === "good" ? accent : b.tone === "bad" ? "#c2456b" : "#8aa0c0";
  return (
    <div className="exp-bar">
      <div className="exp-bar-head">
        <span className="exp-bar-label">
          {b.winner && <span className="exp-bar-star" aria-hidden>★</span>}
          {b.label}
        </span>
        <span className="exp-bar-value">{b.value}</span>
      </div>
      <div className="exp-bar-track">
        <div
          className={clsx("exp-bar-fill", b.winner && "is-winner")}
          style={{ width: shown ? `${b.pct}%` : "0%", background: color }}
        />
      </div>
      {b.sub && <span className="exp-bar-sub">{b.sub}</span>}
    </div>
  );
}

function MetricBody({ view, accent }: { view: Extract<View, { kind: "metric" }>; accent: string }) {
  const [metric, setMetric] = useState(view.series[0]?.key ?? "");
  const [shown, setShown] = useState(true);
  const series = view.series.find((s) => s.key === metric) ?? view.series[0];
  return (
    <div>
      <div className="exp-metric-switch" role="tablist">
        {view.series.map((s) => (
          <button
            key={s.key}
            role="tab"
            aria-selected={s.key === series.key}
            className={clsx("exp-metric-btn", s.key === series.key && "is-active")}
            style={s.key === series.key ? { background: accent, borderColor: accent } : undefined}
            onClick={() => {
              setShown(false);
              setMetric(s.key);
              requestAnimationFrame(() => setShown(true));
            }}
          >
            {s.label}
          </button>
        ))}
      </div>
      <div className="exp-bars">
        {series.bars.map((b) => (
          <BarRow key={b.label} b={b} accent={accent} shown={shown} />
        ))}
      </div>
    </div>
  );
}

function ViewBody({ view, accent, shown }: { view: View; accent: string; shown: boolean }) {
  if (view.kind === "metric") {
    return <MetricBody view={view} accent={accent} />;
  }
  if (view.kind === "findings") {
    return (
      <div className="exp-findings">
        {view.findings.map((f, i) => (
          <div key={i} className="exp-finding">
            {f.stat && <span className="exp-finding-stat" style={{ color: accent }}>{f.stat}</span>}
            <div className="exp-finding-body">
              <h4 className="exp-finding-title">
                {f.id && <span className="exp-finding-id">{f.id}</span>}
                {f.title}
              </h4>
              <p>{f.text}</p>
            </div>
          </div>
        ))}
      </div>
    );
  }
  if (view.kind === "bars") {
    return (
      <div>
        <p className="exp-metric-label">{view.metricLabel}</p>
        <div className="exp-bars">
          {view.bars.map((b) => (
            <BarRow key={b.label} b={b} accent={accent} shown={shown} />
          ))}
        </div>
        {view.caption && <p className="exp-caption">{view.caption}</p>}
      </div>
    );
  }
  return (
    <div>
      <div className="exp-table-wrap">
        <table className="exp-table">
          <thead>
            <tr>
              {view.columns.map((c, i) => (
                <th key={c} className={i === 0 ? "exp-th-label" : undefined}>
                  {c}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {view.rows.map((row) => (
              <tr key={row[0]}>
                {row.map((cell, i) => (
                  <td key={i} className={i === 0 ? "exp-td-label" : undefined}>
                    {cell}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {view.caption && <p className="exp-caption">{view.caption}</p>}
    </div>
  );
}

export function ResultGroupViz({ group, accent }: { group: ResultGroup; accent: string }) {
  const [active, setActive] = useState(group.views[0]?.key ?? "");
  const [shown, setShown] = useState(false);

  useEffect(() => {
    const t = setTimeout(() => setShown(true), 60);
    return () => clearTimeout(t);
  }, []);

  const view = group.views.find((v) => v.key === active) ?? group.views[0];
  const tabbed = group.views.length > 1;

  return (
    <section className="exp-group">
      <h3 className="exp-group-title">{group.title}</h3>
      {group.intro && <p className="exp-group-intro">{group.intro}</p>}
      {tabbed && (
        <div className="exp-tabs" role="tablist">
          {group.views.map((v) => (
            <button
              key={v.key}
              role="tab"
              aria-selected={v.key === view.key}
              className={clsx("exp-tab", v.key === view.key && "is-active")}
              style={v.key === view.key ? { borderColor: accent, color: accent } : undefined}
              onClick={() => {
                setShown(false);
                setActive(v.key);
                requestAnimationFrame(() => setShown(true));
              }}
            >
              {v.label}
            </button>
          ))}
        </div>
      )}
      <ViewBody view={view} accent={accent} shown={shown} />
    </section>
  );
}
