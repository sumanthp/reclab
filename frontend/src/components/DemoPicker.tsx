interface Props {
  fixtures: { id: string; label: string }[];
  onSelect: (id: string) => void;
}

export function DemoPicker({ fixtures, onSelect }: Props) {
  return (
    <div className="upload-card">
      <h1>Try a real dataset</h1>
      <p className="intro">
        Pick one below to see the actual profile, shortlist, and full architecture comparison —
        real results, not mocked-up numbers. Each one is a run this project's own benchmark
        script produced against real data.
      </p>
      <div className="demo-picker-list">
        {fixtures.map((f) => (
          <button key={f.id} type="button" className="demo-picker-item" onClick={() => onSelect(f.id)}>
            {f.label}
          </button>
        ))}
      </div>
    </div>
  );
}
