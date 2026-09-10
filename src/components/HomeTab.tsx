import { useState, useCallback } from "react";
import { useQuery } from "@tanstack/react-query";
import ChatTab from "./ChatTab";
import { LocalAiView } from "../views/LocalAiView";
import { useAgents } from "../hooks/useAgents";
import { agentStatusColor, agentStatusLabel } from "../utils/agentStatus";
import { useAppContext } from "../context/AppContext";
import "../ui/local-ai/local-ai.css";

const DEFAULT_SUGGESTIONS = [
  "What's on my screen?",
  "Summarize this document",
  "Help me debug this code",
  "Write a professional email",
];

function AgentDockStrip() {
  const { agents, loading } = useAgents();
  const { setActiveTab } = useAppContext();

  if (loading || agents.length === 0) return null;

  return (
    <div className="agent-dock-strip" role="list" aria-label="Active agents">
      <span className="agent-dock-label">Agents</span>
      <div className="agent-dock-items">
        {agents.slice(0, 6).map((agent) => {
          const label = agentStatusLabel(agent.state);
          const color = agentStatusColor(agent.state);
          return (
            <div
              key={agent.slug}
              className="agent-dock-item"
              title={`${agent.name} (${label})`}
              role="listitem"
              onClick={() => setActiveTab("agents")}
              style={{ cursor: "pointer" }}
            >
              <span className="agent-dock-dot" style={{ backgroundColor: color }} />
              <span className="agent-dock-name">{agent.name}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// F-027: Empty-state CTA when no agents exist
function EmptyAgentsCTA({ onCreateAgent }: { onCreateAgent: () => void }) {
  return (
    <div className="empty-agents-cta">
      <div className="empty-icon">✦</div>
      <h3>Create your first agent</h3>
      <p>Agents can automate tasks, answer questions, and control your computer.</p>
      <button className="btn-primary" onClick={onCreateAgent}>
        Create Agent
      </button>
    </div>
  );
}

function HomeTab() {
  const [showChat, setShowChat] = useState(true);
  const [showDetect, setShowDetect] = useState(false);
  const [initialSuggestion, setInitialSuggestion] = useState<string | null>(null);
  const { agents, loading: agentsLoading } = useAgents();
  const { setActiveTab } = useAppContext();

  // F-026: Dynamic suggestions from recent prompts
  const { data: suggestions = DEFAULT_SUGGESTIONS } = useQuery({
    queryKey: ["home-suggestions"],
    queryFn: async (): Promise<string[]> => {
      try {
        const recent = JSON.parse(sessionStorage.getItem("recent_prompts") || "[]") as string[];
        return recent.length > 0 ? recent.slice(0, 4) : DEFAULT_SUGGESTIONS;
      } catch {
        return DEFAULT_SUGGESTIONS;
      }
    },
    staleTime: 60_000,
  });

  const handleSuggestion = useCallback((suggestion: string) => {
    setInitialSuggestion(suggestion);
    setShowChat(true);

    // Record this prompt for future suggestions
    try {
      const recent = JSON.parse(sessionStorage.getItem("recent_prompts") || "[]") as string[];
      const updated = [suggestion, ...recent.filter((s) => s !== suggestion)].slice(0, 20);
      sessionStorage.setItem("recent_prompts", JSON.stringify(updated));
    } catch {
      // non-fatal
    }
  }, []);

  if (showDetect) {
    return (
      <div className="home-tab">
        <button className="home-back-btn" onClick={() => setShowDetect(false)} aria-label="Back to home">
          ← Back
        </button>
        <LocalAiView initialScreen="detect" />
      </div>
    );
  }

  if (showChat) {
    return (
      <div className="home-tab">
        <div className="lai-row-spread">
          <button className="home-back-btn" onClick={() => setShowChat(false)} aria-label="Back to home">
            ← Back
          </button>
          <button
            className="home-back-btn"
            onClick={() => {
              setShowChat(false);
              setShowDetect(true);
            }}
            aria-label="Open AI image detection"
          >
            Detect AI
          </button>
        </div>
        <ChatTab initialText={initialSuggestion ?? undefined} />
      </div>
    );
  }

  return (
    <div className="home-tab">
      <AgentDockStrip />

      {/* F-027: Empty agents CTA */}
      {!agentsLoading && agents.length === 0 && (
        <EmptyAgentsCTA onCreateAgent={() => setActiveTab("agents")} />
      )}

      <div className="hero-card">
        <h1>Hi, I'm ClickyX</h1>
        <p>Your AI companion — ask me anything about your screen.</p>
      </div>
      <button className="start-chat-btn" onClick={() => setShowChat(true)}>
        Start a conversation
      </button>
      <button className="start-chat-btn" onClick={() => setShowDetect(true)}>
        Detect AI images
      </button>

      {/* F-026: Dynamic suggestion chips */}
      <div className="suggestions-grid">
        {suggestions.map((s) => (
          <button key={s} className="suggestion-chip" onClick={() => handleSuggestion(s)}>
            {s}
          </button>
        ))}
      </div>
    </div>
  );
}

export default HomeTab;
