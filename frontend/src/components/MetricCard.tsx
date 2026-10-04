import type { LucideIcon } from 'lucide-react';

interface MetricCardProps {
  label: string;
  value: string;
  note: string;
  icon: LucideIcon;
  accent?: boolean;
}

export function MetricCard({ label, value, note, icon: Icon, accent = false }: MetricCardProps) {
  return (
    <div className={`metric-card${accent ? ' metric-card--accent' : ''}`}>
      <div className="metric-card__top">
        <span>{label}</span>
        <Icon size={16} strokeWidth={1.8} />
      </div>
      <strong>{value}</strong>
      <small>{note}</small>
    </div>
  );
}
