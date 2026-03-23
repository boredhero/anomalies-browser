import { useEffect } from 'react';
import { Sparkles } from 'lucide-react';
import type { Detection } from '../../types';
import { FEATURE_LABELS } from '../../types';

interface ResultsSplashProps {
  detections: Detection[];
  onDismiss: () => void;
  autoMs?: number;
}

export default function ResultsSplash({ detections, onDismiss, autoMs = 3000 }: ResultsSplashProps) {
  // Auto-dismiss after autoMs
  useEffect(() => {
    const timer = setTimeout(onDismiss, autoMs);
    return () => clearTimeout(timer);
  }, [onDismiss, autoMs]);

  // Aggregate by type
  const typeCounts: Record<string, number> = {};
  for (const d of detections) {
    const t = d.feature_type || 'unknown';
    typeCounts[t] = (typeCounts[t] || 0) + 1;
  }

  const topTypes = Object.entries(typeCounts)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 3)
    .map(([t, c]) => `${c} ${FEATURE_LABELS[t as keyof typeof FEATURE_LABELS] || t}${c > 1 ? 's' : ''}`)
    .join(', ');

  return (
    <div
      className="fixed inset-0 z-50 bg-slate-950/90 backdrop-blur flex items-center justify-center cursor-pointer"
      onClick={onDismiss}
    >
      <div className="text-center px-8 animate-in fade-in zoom-in duration-500">
        <Sparkles size={48} className="text-yellow-400 mx-auto mb-4" />
        <h2 className="text-4xl font-black text-white mb-3">
          Found <span className="text-hotpink-400">{detections.length}</span> potential features!
        </h2>
        {topTypes && (
          <p className="text-lg text-slate-400 mb-6">
            Including {topTypes}
          </p>
        )}
        <p className="text-sm text-slate-600">Tap anywhere to explore</p>
      </div>
    </div>
  );
}
