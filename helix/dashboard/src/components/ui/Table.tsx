'use client';

import { cn } from '@/lib/utils';

interface Column {
  key: string;
  label: string;
  align?: 'left' | 'right' | 'center';
  mono?: boolean;
}

interface TableProps {
  columns: Column[];
  data: Record<string, any>[];
  onRowClick?: (row: Record<string, any>) => void;
  className?: string;
}

const alignClass: Record<string, string> = {
  left: 'text-left',
  right: 'text-right',
  center: 'text-center',
};

export function Table({ columns, data, onRowClick, className }: TableProps) {
  return (
    <div className={cn('w-full overflow-x-auto', className)}>
      <table className="w-full border-collapse">
        <thead>
          <tr>
            {columns.map((col) => (
              <th
                key={col.key}
                className={cn(
                  'px-4 py-3.5 text-sm font-medium text-helix-muted',
                  alignClass[col.align ?? 'left'],
                )}
              >
                {col.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.map((row, rowIndex) => (
            <tr
              key={rowIndex}
              onClick={onRowClick ? () => onRowClick(row) : undefined}
              className={cn(
                'border-b border-helix-border',
                onRowClick && 'cursor-pointer hover:bg-white/[0.02] transition-colors',
              )}
            >
              {columns.map((col) => (
                <td
                  key={col.key}
                  className={cn(
                    'px-4 py-3.5 text-base text-helix-text2',
                    alignClass[col.align ?? 'left'],
                    col.mono && 'font-mono',
                  )}
                >
                  {row[col.key]}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
