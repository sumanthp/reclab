import type { DataProfile } from "../lib/types";
import { int, pct } from "../lib/format";

export function ProfileStats({ profile }: { profile: DataProfile }) {
  return (
    <div className="stat-grid">
      <Tile label="Users" value={int(profile.n_users)} />
      <Tile label="Items" value={int(profile.n_items)} />
      <Tile label="Interactions" value={int(profile.n_interactions)} />
      <Tile label="Sparsity" value={pct(profile.sparsity)} />
      <Tile label="Cold-start ratio" value={pct(profile.cold_start_ratio)} />
      <Tile label="Median sequence len." value={profile.median_sequence_length.toFixed(1)} />
      <BoolTile label="Item text" value={profile.has_item_text} />
      <BoolTile label="Item image" value={profile.has_item_image} />
    </div>
  );
}

function Tile({ label, value }: { label: string; value: string }) {
  return (
    <div className="stat-tile">
      <div className="stat-label">{label}</div>
      <div className="stat-value">{value}</div>
    </div>
  );
}

function BoolTile({ label, value }: { label: string; value: boolean }) {
  return (
    <div className="stat-tile bool">
      <div className="stat-label">{label}</div>
      <div className={`stat-value ${value ? "yes" : "no"}`}>
        {value ? "✓ present" : "— absent"}
      </div>
    </div>
  );
}
