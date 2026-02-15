import { cn } from "../../lib/utils";

interface Column<T> {
  key: string;
  header: string;
  render: (item: T) => React.ReactNode;
  className?: string;
}

interface DataTableProps<T> {
  columns: Column<T>[];
  data: T[];
  keyFn: (item: T) => string;
  className?: string;
  emptyMessage?: string;
}

export function DataTable<T>({
  columns,
  data,
  keyFn,
  className,
  emptyMessage = "No data",
}: DataTableProps<T>) {
  return (
    <div className={cn("overflow-auto", className)}>
      <table className="w-full text-mono-data">
        <thead>
          <tr className="border-b border-border-primary">
            {columns.map((col) => (
              <th
                key={col.key}
                className={cn(
                  "text-left text-label text-text-tertiary py-1 px-1.5 font-medium",
                  col.className
                )}
              >
                {col.header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.length === 0 ? (
            <tr>
              <td
                colSpan={columns.length}
                className="text-center text-text-tertiary py-4 text-body"
              >
                {emptyMessage}
              </td>
            </tr>
          ) : (
            data.map((item) => (
              <tr
                key={keyFn(item)}
                className="border-b border-border-grid hover:bg-bg-hover transition-colors"
              >
                {columns.map((col) => (
                  <td
                    key={col.key}
                    className={cn(
                      "py-1 px-1.5 text-text-secondary",
                      col.className
                    )}
                  >
                    {col.render(item)}
                  </td>
                ))}
              </tr>
            ))
          )}
        </tbody>
      </table>
    </div>
  );
}
