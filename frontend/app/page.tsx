"use client";

import { useEffect, useState } from "react";
import { ArrowRight, Github, MessageSquare, FlaskConical } from "lucide-react";
import { EXPERIMENTS, getExperiment } from "@/lib/experiments";
import { ScaleSection } from "@/components/scale-section";
import { ExperimentView } from "@/components/experiment-view";

const REPO = "https://github.com/towardsai/ai-tutor-app";
const TUTOR_EMBED = "https://towardsai-tutors-ai-tutor-chatbot.hf.space";

// Hash routing (#slm, #gemini, ...). Everything is served from the single
// index.html, which is the only path an HF Static Space serves reliably for a
// multi-view site; the hash switches the view client-side and is deep-linkable.
export default function Page() {
  const [slug, setSlug] = useState("");
  useEffect(() => {
    const read = () => setSlug(window.location.hash.replace(/^#\/?/, ""));
    read();
    window.addEventListener("hashchange", read);
    return () => window.removeEventListener("hashchange", read);
  }, []);
  useEffect(() => {
    window.scrollTo(0, 0);
  }, [slug]);

  const exp = getExperiment(slug);
  return exp ? <ExperimentView exp={exp} /> : <HomeView />;
}

function HomeView() {
  const experiments = [...EXPERIMENTS].sort((a, b) => a.order - b.order);
  return (
    <main className="home">
      <div className="home-wrap">
        <header className="hero">
          <span className="hero-kicker">Towards AI · Workshop</span>
          <h1 className="hero-title">
            Context Engineering in 2026: Compaction, Memory &amp; Cost
          </h1>
          <p className="hero-lead">
            Every long agent session eventually breaks: the assistant that swore it
            would never push to main does exactly that forty turns later. The model
            did not get dumber, its context did. Context engineering is deciding what
            the model sees on every single call (instructions, history, retrieved
            course content, memory, and tool outputs), and it is the line between a
            tutor that holds a coherent session and one that forgets the student
            halfway through.
          </p>
          <p className="hero-lead">
            We show it with Towards AI&rsquo;s open-source AI tutor for our
            AI-engineering courses: the compaction toolkit, memory that survives
            across sessions, and production retrieval, each measured on Gemini for
            tokens, cost, latency, and memory probes instead of vibe-checks. At real
            volume even Gemini Flash got expensive, so we tested whether open and
            local models match the quality for a fraction of the cost. Everything is
            open source.
          </p>
          <div className="hero-actions">
            <a href="#tutor" className="btn btn-primary">
              <MessageSquare size={17} /> Try the tutor
            </a>
            <a href="#experiments" className="btn">
              <FlaskConical size={17} /> See experiments
            </a>
            <a href={REPO} target="_blank" rel="noreferrer" className="btn btn-ghost">
              <Github size={17} /> Repository
            </a>
          </div>
        </header>

        <section id="tutor" className="tutor-section">
          <div className="section-head">
            <h2>Try the AI tutor</h2>
          </div>
          <p className="section-intro">
            The live production tutor, embedded below. Ask it about RAG, agents,
            embeddings, or anything from the course corpus. It grounds answers in a
            curated knowledge base and shows its sources.
          </p>
          <div className="tutor-frame">
            <iframe src={TUTOR_EMBED} title="AI Tutor" loading="lazy" allow="clipboard-write" />
          </div>
        </section>

        <section id="scale" className="scale-section">
          <div className="section-head">
            <h2>The problem, by the numbers</h2>
          </div>
          <p className="section-intro">
            Context engineering exists because of these magnitudes: a finite window, a
            long lesson, retrieval payloads that dwarf the chat, and sessions that grow
            for dozens of turns.
          </p>
          <ScaleSection />
        </section>

        <section id="experiments" className="experiments-section">
          <div className="section-head">
            <h2>Experiments and results</h2>
          </div>
          <p className="section-intro">
            Each study isolates one question and reports it on the same lesson and
            question set. Open any card for the interactive results.
          </p>
          <div className="exp-grid">
            {experiments.map((e) => (
              <a key={e.slug} href={`#${e.slug}`} className="exp-card">
                <div className="exp-card-accent" style={{ background: e.accent }} />
                <span className="exp-card-badge">{e.badge}</span>
                <h3 className="exp-card-title">{e.shortTitle}</h3>
                <p className="exp-card-q">{e.question}</p>
                <p className="exp-card-takeaway">{e.takeaway}</p>
                <span className="exp-card-cta" style={{ color: e.accent }}>
                  View results <ArrowRight size={15} />
                </span>
              </a>
            ))}
          </div>
        </section>

        <footer className="home-footer">
          <span>Towards AI · AI tutor workshop</span>
          <a href={REPO} target="_blank" rel="noreferrer">
            github.com/towardsai/ai-tutor-app
          </a>
        </footer>
      </div>
    </main>
  );
}
