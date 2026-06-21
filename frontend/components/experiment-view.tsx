"use client";

import { ArrowLeft, ExternalLink } from "lucide-react";
import type { Experiment } from "@/lib/experiments";
import { ResultGroupViz } from "@/components/experiment-viz";

export function ExperimentView({ exp }: { exp: Experiment }) {
  return (
    <main className="exp-page" style={{ ["--exp-accent" as string]: exp.accent }}>
      <div className="exp-wrap">
        <a href="#" className="exp-back">
          <ArrowLeft size={16} /> All experiments
        </a>

        <header className="exp-header">
          <span className="exp-badge">{exp.badge}</span>
          <h1 className="exp-title">{exp.title}</h1>
          <p className="exp-question">{exp.question}</p>
          <div className="exp-takeaway">
            <span className="exp-takeaway-tag">Takeaway</span>
            <p>{exp.takeaway}</p>
          </div>
          <div className="exp-chips">
            {exp.highlights.map((h) => (
              <div key={h.label} className="exp-chip">
                <span className="exp-chip-label">{h.label}</span>
                <span className="exp-chip-value">{h.value}</span>
              </div>
            ))}
          </div>
        </header>

        {exp.groups.map((g) => (
          <ResultGroupViz key={g.title} group={g} accent={exp.accent} />
        ))}

        <details className="exp-details">
          <summary>How it worked</summary>
          <ul>
            {exp.setup.map((s, i) => (
              <li key={i}>{s}</li>
            ))}
          </ul>
        </details>

        <details className="exp-details">
          <summary>Caveats</summary>
          <ul>
            {exp.caveats.map((c, i) => (
              <li key={i}>{c}</li>
            ))}
          </ul>
        </details>

        <div className="exp-links">
          {exp.links.map((l) => (
            <a key={l.href} href={l.href} target="_blank" rel="noreferrer" className="exp-link">
              {l.label} <ExternalLink size={14} />
            </a>
          ))}
        </div>
      </div>
    </main>
  );
}
