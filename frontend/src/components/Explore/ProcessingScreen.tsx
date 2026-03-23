import { useState, useEffect } from 'react';
import { Check, Loader2, AlertCircle } from 'lucide-react';

const STAGES = [
  { key: 'discovering', label: 'Downloading terrain data...' },
  { key: 'downloading', label: 'Downloading terrain data...' },
  { key: 'analyzing', label: 'Detecting underground features...' },
  { key: 'finishing', label: 'Almost done!' },
];

const FUN_FACTS = [
  'LiDAR can see through tree canopy to reveal hidden terrain features',
  'Sinkholes form when underground limestone dissolves over thousands of years',
  'The deepest cave in the US is Lechuguilla Cave in New Mexico — 1,604 feet deep',
  'Pennsylvania has over 1,000 documented caves',
  'Some sinkholes open suddenly and can swallow entire buildings',
  'Abandoned mine portals can be detected by their distinctive terrain signatures',
  'LiDAR stands for Light Detection and Ranging — it uses laser pulses to map the Earth',
  'The longest cave system in the world is Mammoth Cave in Kentucky at 426 miles',
  'Karst terrain covers about 20% of the Earth\'s land surface',
];

function getStageIndex(stage: string | null, progress: number): number {
  if (!stage) {
    if (progress < 10) return 0;
    if (progress < 40) return 1;
    if (progress < 90) return 2;
    return 3;
  }
  const idx = STAGES.findIndex((s) => s.key === stage);
  return idx >= 0 ? idx : 0;
}

interface ProcessingScreenProps {
  progress: number;
  stage: string | null;
  error: string | null;
  onRetry: () => void;
}

export default function ProcessingScreen({ progress, stage, error, onRetry }: ProcessingScreenProps) {
  const [factIndex, setFactIndex] = useState(0);
  const [factVisible, setFactVisible] = useState(true);

  // Rotate fun facts every 8 seconds
  useEffect(() => {
    const interval = setInterval(() => {
      setFactVisible(false);
      setTimeout(() => {
        setFactIndex((i) => (i + 1) % FUN_FACTS.length);
        setFactVisible(true);
      }, 400);
    }, 8000);
    return () => clearInterval(interval);
  }, []);

  if (error) {
    return (
      <div className="fixed inset-0 z-50 bg-slate-950 flex items-center justify-center">
        <div className="text-center px-8 max-w-md">
          <AlertCircle size={48} className="text-red-400 mx-auto mb-4" />
          <h2 className="text-xl font-bold text-white mb-2">Something went wrong</h2>
          <p className="text-slate-400 mb-6">{error}</p>
          <button
            onClick={onRetry}
            className="bg-blue-600 hover:bg-blue-500 text-white font-medium px-6 py-3 rounded transition-colors"
          >
            Try again
          </button>
        </div>
      </div>
    );
  }

  const currentStageIdx = getStageIndex(stage, progress);
  const currentLabel = STAGES[currentStageIdx]?.label || 'Scanning your area...';

  return (
    <div className="fixed inset-0 z-50 bg-slate-950 flex items-center justify-center">
      <div className="text-center px-8 max-w-lg w-full">
        {/* Spinning icon */}
        <Loader2 size={40} className="animate-spin text-blue-400 mx-auto mb-6" />

        {/* Stage label */}
        <h2 className="text-xl font-bold text-white mb-6">{currentLabel}</h2>

        {/* Stage indicators */}
        <div className="flex items-center justify-center gap-3 mb-8">
          {STAGES.map((s, i) => {
            const isComplete = i < currentStageIdx;
            const isCurrent = i === currentStageIdx;
            return (
              <div key={s.key} className="flex items-center gap-3">
                <div
                  className={`w-8 h-8 rounded flex items-center justify-center text-sm font-medium transition-colors ${
                    isComplete
                      ? 'bg-green-600 text-white'
                      : isCurrent
                        ? 'bg-blue-600 text-white'
                        : 'bg-slate-800 text-slate-500'
                  }`}
                >
                  {isComplete ? <Check size={16} /> : i + 1}
                </div>
                {i < STAGES.length - 1 && (
                  <div className={`w-8 h-0.5 ${i < currentStageIdx ? 'bg-green-600' : 'bg-slate-700'}`} />
                )}
              </div>
            );
          })}
        </div>

        {/* Progress bar */}
        <div className="w-full h-2 bg-slate-800 rounded overflow-hidden mb-3">
          <div
            className="h-full bg-blue-500 rounded transition-all duration-700 ease-out"
            style={{ width: `${Math.max(progress, 2)}%` }}
          />
        </div>
        <p className="text-sm text-slate-400 font-mono mb-10">{Math.round(progress)}%</p>

        {/* Fun fact */}
        <div className="h-16 flex items-center justify-center">
          <p
            className={`text-sm text-slate-500 italic max-w-sm transition-opacity duration-300 ${
              factVisible ? 'opacity-100' : 'opacity-0'
            }`}
          >
            {FUN_FACTS[factIndex]}
          </p>
        </div>
      </div>
    </div>
  );
}
