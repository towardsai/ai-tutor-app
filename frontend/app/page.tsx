import Link from "next/link";
import { ArrowRight, Github, MessageSquare, FlaskConical } from "lucide-react";
import { EXPERIMENTS } from "@/lib/experiments";

const REPO = "https://github.com/towardsai/ai-tutor-app";
const TUTOR_EMBED = "https://towardsai-tutors-ai-tutor-chatbot.hf.space";

export default function Home() {
  const experiments = [...EXPERIMENTS].sort((a, b) => a.order - b.order);
  return (
    <main className="home">
      <div className="home-wrap">
        <header className="hero">
          <span className="hero-kicker">Towards AI · Workshop</span>
          <h1 className="hero-title">
            Filling the context window: what actually works for an AI tutor
          </h1>
          <p className="hero-lead">
            A retrieval-grounded AI tutor for applied AI, LLMs, and RAG, plus the
            experiments behind it. We measured how to put long course content into
            the model context, across large cloud models and cheap local ones, on
            answer quality, tokens, cost, and latency. Try the tutor below, then
            dig into each experiment and its results.
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
            <h2>The AI tutor, live</h2>
            <Link href="/chat" className="section-link">
              Open the local build <ArrowRight size={15} />
            </Link>
          </div>
          <p className="section-intro">
            The production tutor running on Hugging Face. Ask it about RAG, agents,
            embeddings, or anything from the course corpus. It grounds answers in a
            curated knowledge base and shows its sources.
          </p>
          <div className="tutor-frame">
            <iframe
              src={TUTOR_EMBED}
              title="AI Tutor"
              loading="lazy"
              allow="clipboard-write"
            />
          </div>
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
              <Link key={e.slug} href={`/experiments/${e.slug}`} className="exp-card">
                <div className="exp-card-accent" style={{ background: e.accent }} />
                <span className="exp-card-badge">{e.badge}</span>
                <h3 className="exp-card-title">{e.shortTitle}</h3>
                <p className="exp-card-q">{e.question}</p>
                <p className="exp-card-takeaway">{e.takeaway}</p>
                <span className="exp-card-cta" style={{ color: e.accent }}>
                  View results <ArrowRight size={15} />
                </span>
              </Link>
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
