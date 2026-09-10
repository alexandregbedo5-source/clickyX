export function LocalAiHelpScreen() {
  return (
    <section className="lai-card lai-help" aria-labelledby="lai-help-title">
      <h3 id="lai-help-title" className="lai-title">How this works</h3>

      <h4>Offline mode</h4>
      <p>
        The status chip in the bar tells you if ClickyX can reach the Internet.
        <strong> Local only</strong> means cloud chat is blocked and the app
        should use Ollama / system voice instead.
      </p>
      <ul>
        <li>Install Ollama and pull a model once while you still have Internet.</li>
        <li>Optional: run a local Whisper server for dictation without the cloud.</li>
        <li>Use Settings → Offline to force offline mode or change local URLs.</li>
      </ul>

      <h4>AI image detection</h4>
      <p>
        Images are sent only to the local detector (<code>POST /detect-ai-image</code>
        on <code>127.0.0.1:32188</code>). Start it with{" "}
        <code>python -m ai_detector serve</code> from the detector branch.
      </p>
      <ul>
        <li><strong>Likely AI-generated</strong> — confidence is above the engine threshold (0.5 by default).</li>
        <li><strong>Likely a photograph</strong> — confidence is below that threshold.</li>
        <li>Scores near 50% are marked uncertain. This is an estimate, not proof.</li>
      </ul>

      <h4>History</h4>
      <p>
        Past verdicts stay in local storage on this machine. Clear them anytime
        from the History screen. Nothing is synced.
      </p>

      <h4>If something is missing</h4>
      <ul>
        <li>Detector unreachable → start the Python service, then refresh.</li>
        <li>Offline APIs missing → merge or run the offline-engine branch.</li>
        <li>Ollama not found → <code>ollama serve</code> then <code>ollama pull llama3.2</code>.</li>
      </ul>
    </section>
  );
}

export default LocalAiHelpScreen;
