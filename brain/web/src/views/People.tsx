import { useState } from "react";
import { Link } from "react-router-dom";
import { Check, GitMerge, Pencil, Search, Trash2, UserPlus } from "lucide-react";
import {
  deletePerson,
  getPeople,
  mergePerson,
  renamePerson,
} from "@/api/client";
import type { Person } from "@/api/types";
import { useAsync } from "@/hooks/useAsync";
import { Card } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { Input, Select } from "@/components/ui/Input";
import { Modal } from "@/components/ui/Modal";
import {
  CenteredSpinner,
  EmptyState,
  ErrorState,
  Skeleton,
} from "@/components/ui/States";
import { PersonAvatar } from "@/components/PersonAvatar";
import { fmtDateTime, isUnknown, timeAgo } from "@/lib/utils";

export default function People() {
  const { data, loading, error, refetch } = useAsync(getPeople, []);
  const [query, setQuery] = useState("");
  const [filter, setFilter] = useState<"all" | "known" | "unknown">("all");
  const [busy, setBusy] = useState(false);

  const [renaming, setRenaming] = useState<Person | null>(null);
  const [renameValue, setRenameValue] = useState("");
  const [merging, setMerging] = useState<Person | null>(null);
  const [mergeInto, setMergeInto] = useState<number | "">("");
  const [deleting, setDeleting] = useState<Person | null>(null);

  const people = data ?? [];
  const filtered = people.filter((p) => {
    const matchQuery =
      !query || p.label.toLowerCase().includes(query.toLowerCase());
    const matchFilter =
      filter === "all" ||
      (filter === "known" && !isUnknown(p)) ||
      (filter === "unknown" && isUnknown(p));
    return matchQuery && matchFilter;
  });

  async function doRename() {
    if (!renaming) return;
    setBusy(true);
    try {
      await renamePerson(renaming.id, renameValue.trim());
      setRenaming(null);
      refetch();
    } finally {
      setBusy(false);
    }
  }

  async function doMerge() {
    if (!merging || mergeInto === "" || mergeInto === merging.id) return;
    setBusy(true);
    try {
      await mergePerson(merging.id, Number(mergeInto));
      setMerging(null);
      setMergeInto("");
      refetch();
    } finally {
      setBusy(false);
    }
  }

  async function doDelete() {
    if (!deleting) return;
    setBusy(true);
    try {
      await deletePerson(deleting.id);
      setDeleting(null);
      refetch();
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-xl font-semibold tracking-tight text-ink">People</h1>
          <p className="text-sm text-ink-muted">
            Everyone the brain has clustered. Rename, merge duplicates, or delete.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <div className="relative">
            <Search className="pointer-events-none absolute left-2.5 top-1/2 h-4 w-4 -translate-y-1/2 text-ink-faint" />
            <Input
              placeholder="Search…"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              className="w-44 pl-8"
            />
          </div>
          <Select
            value={filter}
            onChange={(e) =>
              setFilter(e.target.value as "all" | "known" | "unknown")
            }
            className="w-32"
          >
            <option value="all">All</option>
            <option value="known">Known</option>
            <option value="unknown">Unknown</option>
          </Select>
        </div>
      </div>

      {loading && !data ? (
        <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-4">
          {Array.from({ length: 8 }).map((_, i) => (
            <Skeleton key={i} className="h-64" />
          ))}
        </div>
      ) : error ? (
        <ErrorState message={error} onRetry={refetch} />
      ) : people.length === 0 ? (
        <EmptyState
          icon={<UserPlus className="h-6 w-6" />}
          title="No people yet"
          hint="As the camera sees faces, the brain will cluster them into people here. Start the pipeline and let it watch the room."
        />
      ) : filtered.length === 0 ? (
        <EmptyState
          icon={<Search className="h-6 w-6" />}
          title="No matches"
          hint="Try a different search or filter."
        />
      ) : (
        <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-4">
          {filtered.map((p) => (
            <PersonCard
              key={p.id}
              person={p}
              onRename={() => {
                setRenaming(p);
                setRenameValue(p.name ?? "");
              }}
              onMerge={() => {
                setMerging(p);
                setMergeInto("");
              }}
              onDelete={() => setDeleting(p)}
            />
          ))}
        </div>
      )}

      {/* Rename modal */}
      <Modal
        open={!!renaming}
        onClose={() => setRenaming(null)}
        title={`Rename ${renaming?.label ?? ""}`}
      >
        <form
          onSubmit={(e) => {
            e.preventDefault();
            void doRename();
          }}
          className="space-y-4"
        >
          <Input
            autoFocus
            placeholder="Name (e.g. Sam)"
            value={renameValue}
            onChange={(e) => setRenameValue(e.target.value)}
          />
          <p className="text-xs text-ink-muted">
            Naming a person marks them as <b className="text-ink">known</b>.
          </p>
          <div className="flex justify-end gap-2">
            <Button type="button" variant="ghost" onClick={() => setRenaming(null)}>
              Cancel
            </Button>
            <Button
              type="submit"
              variant="primary"
              disabled={busy || !renameValue.trim()}
            >
              <Check className="h-4 w-4" /> Save
            </Button>
          </div>
        </form>
      </Modal>

      {/* Merge modal */}
      <Modal
        open={!!merging}
        onClose={() => setMerging(null)}
        title={`Merge ${merging?.label ?? ""} into…`}
      >
        <div className="space-y-4">
          <p className="text-xs text-ink-muted">
            All of {merging?.label}'s faces, sightings and events move into the
            target. {merging?.label} is then deleted. This can't be undone.
          </p>
          <Select
            value={mergeInto}
            onChange={(e) =>
              setMergeInto(e.target.value === "" ? "" : Number(e.target.value))
            }
          >
            <option value="">Select target person…</option>
            {people
              .filter((o) => o.id !== merging?.id)
              .map((o) => (
                <option key={o.id} value={o.id}>
                  {o.label}
                </option>
              ))}
          </Select>
          <div className="flex justify-end gap-2">
            <Button type="button" variant="ghost" onClick={() => setMerging(null)}>
              Cancel
            </Button>
            <Button
              variant="primary"
              disabled={busy || mergeInto === ""}
              onClick={() => void doMerge()}
            >
              <GitMerge className="h-4 w-4" /> Merge
            </Button>
          </div>
        </div>
      </Modal>

      {/* Delete modal */}
      <Modal
        open={!!deleting}
        onClose={() => setDeleting(null)}
        title={`Delete ${deleting?.label ?? ""}?`}
      >
        <div className="space-y-4">
          <p className="text-sm text-ink-muted">
            This permanently erases everything about{" "}
            <b className="text-ink">{deleting?.label}</b> — face vectors, crops,
            sightings and events. (Privacy by design.)
          </p>
          <div className="flex justify-end gap-2">
            <Button type="button" variant="ghost" onClick={() => setDeleting(null)}>
              Cancel
            </Button>
            <Button
              variant="danger"
              disabled={busy}
              onClick={() => void doDelete()}
            >
              <Trash2 className="h-4 w-4" /> Delete
            </Button>
          </div>
        </div>
      </Modal>

      {busy && (
        <div className="fixed bottom-6 right-6 z-50">
          <Card className="flex items-center gap-2 px-4 py-2">
            <CenteredSpinner />
          </Card>
        </div>
      )}
    </div>
  );
}

function PersonCard({
  person,
  onRename,
  onMerge,
  onDelete,
}: {
  person: Person;
  onRename: () => void;
  onMerge: () => void;
  onDelete: () => void;
}) {
  const crops = person.crop_urls ?? [];
  const unknown = isUnknown(person);

  return (
    <Card className="group flex flex-col gap-3 p-3">
      <Link to={`/people/${person.id}`} className="block">
        <div className="relative aspect-square w-full overflow-hidden rounded-xl bg-base-800">
          <PersonAvatar id={person.id} label={person.label} fill />
        </div>
      </Link>

      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0">
          <Link
            to={`/people/${person.id}`}
            className="block truncate text-sm font-semibold text-ink hover:text-accent"
          >
            {person.label}
          </Link>
          <p className="text-[11px] text-ink-faint" title={fmtDateTime(person.last_seen_ts)}>
            seen {timeAgo(person.last_seen_ts)}
          </p>
        </div>
        {unknown ? (
          <Badge variant="amber">unknown</Badge>
        ) : (
          <Badge variant="success">known</Badge>
        )}
      </div>

      {/* crop strip */}
      {crops.length > 1 && (
        <div className="flex gap-1">
          {crops.slice(0, 4).map((url, i) => (
            <img
              key={i}
              src={url}
              alt=""
              loading="lazy"
              className="h-9 w-9 rounded-md border border-line object-cover"
              onError={(e) => {
                (e.currentTarget as HTMLImageElement).style.display = "none";
              }}
            />
          ))}
        </div>
      )}

      <div className="mt-auto flex items-center gap-1 opacity-70 transition-opacity group-hover:opacity-100">
        <Button variant="ghost" size="sm" className="flex-1" onClick={onRename}>
          <Pencil className="h-3.5 w-3.5" /> Rename
        </Button>
        <Button variant="ghost" size="icon" onClick={onMerge} title="Merge">
          <GitMerge className="h-4 w-4" />
        </Button>
        <Button
          variant="ghost"
          size="icon"
          onClick={onDelete}
          title="Delete"
          className="text-accent-rose hover:bg-accent-rose/10"
        >
          <Trash2 className="h-4 w-4" />
        </Button>
      </div>
    </Card>
  );
}
