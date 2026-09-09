import {
  DetectHistoryScreen,
  DetectResultsScreen,
  DetectScreen,
  LocalAiHelpScreen,
  LocalAiSettingsScreen,
} from "../pages";
import { LocalAiProvider, useLocalAi } from "../ui/local-ai";
import type { LocalAiScreen } from "../ui/local-ai";
import "../ui/local-ai/local-ai.css";

const NAV: { id: LocalAiScreen; label: string }[] = [
  { id: "detect", label: "Detect" },
  { id: "results", label: "Results" },
  { id: "history", label: "History" },
  { id: "settings", label: "Settings" },
  { id: "help", label: "Help" },
];

function LocalAiShell() {
  const { screen, setScreen } = useLocalAi();

  return (
    <div className="lai-view" data-testid="local-ai-view">
      <nav className="lai-nav" aria-label="Local AI">
        {NAV.map((item) => (
          <button
            key={item.id}
            type="button"
            className={`lai-nav-btn${screen === item.id ? " active" : ""}`}
            aria-current={screen === item.id ? "page" : undefined}
            onClick={() => setScreen(item.id)}
          >
            {item.label}
          </button>
        ))}
      </nav>
      {screen === "detect" && <DetectScreen />}
      {screen === "results" && <DetectResultsScreen />}
      {screen === "history" && <DetectHistoryScreen />}
      {screen === "settings" && <LocalAiSettingsScreen />}
      {screen === "help" && <LocalAiHelpScreen />}
    </div>
  );
}

interface Props {
  initialScreen?: LocalAiScreen;
}

export function LocalAiView({ initialScreen = "detect" }: Props) {
  return (
    <LocalAiProvider initialScreen={initialScreen}>
      <LocalAiShell />
    </LocalAiProvider>
  );
}

export default LocalAiView;
