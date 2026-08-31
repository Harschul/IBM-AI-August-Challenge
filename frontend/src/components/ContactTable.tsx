import type { ContactIn } from "../types";

function update<K extends keyof ContactIn>(
  contacts: ContactIn[],
  index: number,
  key: K,
  value: ContactIn[K],
): ContactIn[] {
  return contacts.map((c, i) => (i === index ? { ...c, [key]: value } : c));
}

export function ContactTable({
  contacts,
  onChange,
}: {
  contacts: ContactIn[];
  onChange: (contacts: ContactIn[]) => void;
}) {
  return (
    <div className="contact-table-wrap">
      <table className="contact-table">
        <thead>
          <tr>
            <th>Src</th>
            <th>Dst</th>
            <th>Start (s)</th>
            <th>End (s)</th>
            <th>Rate (bps)</th>
            <th>Prop delay (s)</th>
            <th />
          </tr>
        </thead>
        <tbody>
          {contacts.map((c, i) => (
            <tr key={i}>
              <td>
                <input type="number" value={c.source_id}
                  onChange={(e) => onChange(update(contacts, i, "source_id", Number(e.target.value)))} />
              </td>
              <td>
                <input type="number" value={c.destination_id}
                  onChange={(e) => onChange(update(contacts, i, "destination_id", Number(e.target.value)))} />
              </td>
              <td>
                <input type="number" value={c.start_s}
                  onChange={(e) => onChange(update(contacts, i, "start_s", Number(e.target.value)))} />
              </td>
              <td>
                <input type="number" value={c.end_s}
                  onChange={(e) => onChange(update(contacts, i, "end_s", Number(e.target.value)))} />
              </td>
              <td>
                <input type="number" value={c.data_rate_bps}
                  onChange={(e) => onChange(update(contacts, i, "data_rate_bps", Number(e.target.value)))} />
              </td>
              <td>
                <input type="number" step="any" value={c.propagation_delay_s}
                  onChange={(e) => onChange(update(contacts, i, "propagation_delay_s", Number(e.target.value)))} />
              </td>
              <td>
                <button
                  type="button"
                  className="row-remove"
                  onClick={() => onChange(contacts.filter((_, j) => j !== i))}
                  aria-label="Remove contact"
                >
                  ×
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <button
        type="button"
        className="row-add"
        onClick={() =>
          onChange([
            ...contacts,
            { source_id: 0, destination_id: 1, start_s: 0, end_s: 100, data_rate_bps: 1_000_000, propagation_delay_s: 0 },
          ])
        }
      >
        + Add contact
      </button>
    </div>
  );
}
