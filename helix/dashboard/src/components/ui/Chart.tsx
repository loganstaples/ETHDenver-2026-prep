'use client';

import {
  ResponsiveContainer,
  LineChart,
  AreaChart,
  Line,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
} from 'recharts';
import { cn } from '@/lib/utils';

interface YKeyConfig {
  key: string;
  label?: string;
}

interface ChartProps {
  data: any[];
  xKey: string;
  yKeys: YKeyConfig[];
  height?: number;
  className?: string;
  type?: 'line' | 'area';
}

const STROKE_OPACITIES = [1.0, 0.4, 0.2];

function CustomTooltip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null;

  return (
    <div className="bg-helix-surface2 border border-helix-border rounded-lg p-2 text-xs shadow-lg">
      <p className="text-helix-muted font-mono mb-1">{label}</p>
      {payload.map((entry: any, i: number) => (
        <p key={i} className="text-helix-text">
          <span className="text-helix-text2">{entry.name}: </span>
          {typeof entry.value === 'number'
            ? entry.value.toLocaleString()
            : entry.value}
        </p>
      ))}
    </div>
  );
}

const AXIS_STYLE = {
  stroke: '#3e3e44',
  tick: { fill: '#63636e', fontSize: 11, fontFamily: 'var(--font-geist-mono)' },
};

export function Chart({
  data,
  xKey,
  yKeys,
  height = 300,
  className,
  type = 'line',
}: ChartProps) {
  const ChartComponent = type === 'area' ? AreaChart : LineChart;

  return (
    <div className={cn('w-full', className)}>
      <ResponsiveContainer width="100%" height={height}>
        <ChartComponent data={data}>
          <defs>
            <linearGradient id="whiteGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="rgba(255,255,255,0.08)" />
              <stop offset="100%" stopColor="rgba(255,255,255,0)" />
            </linearGradient>
          </defs>
          <CartesianGrid
            strokeDasharray="3 3"
            stroke="#1e1e22"
            vertical={false}
          />
          <XAxis
            dataKey={xKey}
            stroke={AXIS_STYLE.stroke}
            tick={AXIS_STYLE.tick}
            tickLine={false}
            axisLine={{ stroke: '#1e1e22' }}
          />
          <YAxis
            stroke={AXIS_STYLE.stroke}
            tick={AXIS_STYLE.tick}
            tickLine={false}
            axisLine={false}
            width={48}
          />
          <RechartsTooltip
            content={<CustomTooltip />}
            cursor={{ stroke: '#2a2a2e', strokeWidth: 1 }}
          />
          {yKeys.map((yKey, i) => {
            const opacity = STROKE_OPACITIES[i] ?? 0.15;
            const name = yKey.label ?? yKey.key;

            if (type === 'area') {
              return (
                <Area
                  key={yKey.key}
                  type="monotone"
                  dataKey={yKey.key}
                  name={name}
                  stroke="#ffffff"
                  strokeOpacity={opacity}
                  strokeWidth={1.5}
                  fill="url(#whiteGradient)"
                  fillOpacity={i === 0 ? 1 : 0}
                  isAnimationActive={true}
                  dot={false}
                />
              );
            }

            return (
              <Line
                key={yKey.key}
                type="monotone"
                dataKey={yKey.key}
                name={name}
                stroke="#ffffff"
                strokeOpacity={opacity}
                strokeWidth={1.5}
                isAnimationActive={true}
                dot={false}
                activeDot={{
                  r: 3,
                  fill: '#ffffff',
                  stroke: '#111113',
                  strokeWidth: 2,
                }}
              />
            );
          })}
        </ChartComponent>
      </ResponsiveContainer>
    </div>
  );
}
