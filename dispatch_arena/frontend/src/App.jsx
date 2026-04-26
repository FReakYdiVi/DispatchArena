// Main app: composes all panels + manages playback state.
//
// State:
//   scenarioId     — which scenario is loaded
//   modeId         — which model's trajectory is being shown ("trained" | "heuristic" | "prompt_only")
//   currentTick    — index into trajectory
//   isPlaying      — autoplay on
//   playbackPhase  — "idle" | "thinking" | "acting"

import React, { useEffect, useMemo, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { SCENARIOS } from './fixtures.js';
import {
  ScenarioMap, OrdersPanel, AgentFeed, RewardPanel,
  ControlBar, TickScrubber, FinalCard, ActionBanner,
} from './components.jsx';
import './styles.css';

const MODES = [
  { id: 'trained',     label: '🤖 trained agent (Qwen3 + GRPO)' },
  { id: 'heuristic',   label: '⚙ greedy heuristic (baseline)' },
  { id: 'prompt_only', label: '💬 untrained LLM (zero-shot)' },
];

const THINKING_MS = 1800;
const ACTION_MS = 1200;

function ScenarioSummary({ scenario }) {
  const m = scenario.metadata;
  return (
    <div className="scenario-summary">
      <div className="scenario-row">
        <span className={`difficulty-badge ${m.difficulty}`}>{m.difficulty}</span>
        <span className="scenario-meta">{m.num_couriers} couriers · {m.num_orders} orders · {m.max_ticks} ticks</span>
      </div>
      <div className="scenario-skills">
        {m.skill_focus.map((s) => <span key={s} className="skill-tag">{s.replace(/_/g, ' ')}</span>)}
      </div>
      <div className="scenario-desc">{m.description}</div>
    </div>
  );
}

function PlaybackPanel({ scenarioId, scenario, modeId, currentTick, onJumpToTick }) {
  const trajectory = scenario.trajectories[modeId] || scenario.trajectories.trained;
  const snapshot = trajectory[Math.min(currentTick, trajectory.length - 1)];
  const isFinished = currentTick >= trajectory.length - 1;

  return (
    <div className="playback-col">
      <div className="map-area">
        <ScenarioMap nodes={scenario.nodes} snapshot={snapshot} />
        {isFinished && <div className="map-final-overlay"><FinalCard trajectory={trajectory} scenario={scenario} /></div>}
      </div>
      <OrdersPanel snapshot={snapshot} maxTicks={scenario.metadata.max_ticks} />
      <RewardPanel snapshot={snapshot} cumulative={snapshot.cumulative_reward ?? 0} />
      <AgentFeed trajectory={trajectory} currentTick={currentTick} onJumpToTick={onJumpToTick} />
    </div>
  );
}

function App() {
  const scenarioIds = Object.keys(SCENARIOS);
  const [scenarioId, setScenarioId] = useState(scenarioIds[0]);
  const [modeId, setModeId] = useState('trained');
  const [currentTick, setCurrentTick] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackPhase, setPlaybackPhase] = useState('idle');

  const scenario = SCENARIOS[scenarioId];
  const trajectory = scenario.trajectories[modeId] || scenario.trajectories.trained;
  const trajectoryLen = useMemo(() => trajectory.length, [trajectory]);

  // Reset tick when scenario or mode changes
  useEffect(() => {
    setCurrentTick(0);
    setIsPlaying(false);
    setPlaybackPhase('idle');
  }, [scenarioId, modeId]);

  // Playback loop: show an explicit "thinking" pause before each action.
  const timeoutRef = useRef(null);
  useEffect(() => {
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
      timeoutRef.current = null;
    }

    if (!isPlaying) {
      setPlaybackPhase('idle');
      return;
    }

    if (currentTick >= trajectoryLen - 1) {
      setIsPlaying(false);
      setPlaybackPhase('idle');
      return;
    }

    setPlaybackPhase('thinking');
    timeoutRef.current = setTimeout(() => {
      setPlaybackPhase('acting');
      timeoutRef.current = setTimeout(() => {
        setCurrentTick((prev) => {
          if (prev >= trajectoryLen - 1) {
            setIsPlaying(false);
            return prev;
          }
          return prev + 1;
        });
      }, ACTION_MS);
    }, THINKING_MS);

    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
        timeoutRef.current = null;
      }
    };
  }, [isPlaying, currentTick, trajectoryLen]);

  const jumpToTick = (tick) => {
    setIsPlaying(false);
    setPlaybackPhase('idle');
    setCurrentTick(tick);
  };

  const stepBack = () => {
    setIsPlaying(false);
    setPlaybackPhase('idle');
    setCurrentTick((t) => Math.max(0, t - 1));
  };

  const stepForward = () => {
    setIsPlaying(false);
    setPlaybackPhase('idle');
    setCurrentTick((t) => Math.min(trajectoryLen - 1, t + 1));
  };

  const resetPlayback = () => {
    setCurrentTick(0);
    setIsPlaying(false);
    setPlaybackPhase('idle');
  };

  return (
    <div className="app">
      <ControlBar
        scenarios={SCENARIOS}
        scenarioId={scenarioId}
        onScenarioChange={setScenarioId}
        modes={MODES}
        modeId={modeId}
        onModeChange={setModeId}
        isPlaying={isPlaying}
        onPlayToggle={() => setIsPlaying((p) => !p)}
        onStepBack={stepBack}
        onStepForward={stepForward}
        onReset={resetPlayback}
      />

      <ScenarioSummary scenario={scenario} />

      <ActionBanner
        trajectory={trajectory}
        currentTick={currentTick}
        playbackPhase={playbackPhase}
        isPlaying={isPlaying}
      />

      <div className="workspace">
        <PlaybackPanel
          scenarioId={scenarioId}
          scenario={scenario}
          modeId={modeId}
          currentTick={currentTick}
          onJumpToTick={jumpToTick}
        />
      </div>

      <TickScrubber trajectory={trajectory}
                    currentTick={currentTick}
                    onTickChange={jumpToTick}
                    maxTicks={scenario.metadata.max_ticks} />
    </div>
  );
}

createRoot(document.getElementById('root')).render(<App />);
