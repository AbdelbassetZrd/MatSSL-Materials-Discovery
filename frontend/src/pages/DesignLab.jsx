import { useState } from "react";
import { Link } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { Lightbulb, Search, Sparkles } from "lucide-react";
import { ErrorBox, Loading, SectionHeader } from "../components/ui.jsx";
import { useComposition } from "../lib/composition.jsx";
import { api } from "../api.js";
import { fmt } from "../lib/theme.js";

// radioactive or strongly toxic elements: the models happily rank them, a chemist would not
const RADIOACTIVE = "Np,Pu,U,Th,Pa,Am,Cm,Ra,Ac,Tc,Po,At,Rn,Fr,Cd,Hg,Tl,Be,As";

const BAR = (v) => (v >= 0.6 ? "bg-good" : v >= 0.3 ? "bg-amber-500" : "bg-ink-mute");
const TEXT = (v) => (v >= 0.6 ? "text-good" : v >= 0.3 ? "text-amber-600" : "text-ink-mute");

function ScoreBar({ value, children }) {
  return (
    <div>
      <div className="flex items-baseline justify-between gap-2">
        <span className="text-sm">{children}</span>
        <b className={`font-mono text-sm ${TEXT(value)}`}>{(100 * value).toFixed(0)}</b>
      </div>
      <div className="mt-1 h-2 rounded bg-surface overflow-hidden">
        <div className={`h-2 rounded ${BAR(value)}`} style={{ width: `${100 * value}%` }} />
      </div>
    </div>
  );
}

function Breakdown({ components }) {
  return (
    <div className="mt-3 space-y-2 border-t border-line pt-3">
      {components.map((c) => (
        <div key={c.label}>
          <div className="flex justify-between text-xs">
            <span className="text-ink-soft">{c.label}</span>
            <span className="font-mono text-ink">{(100 * c.value).toFixed(0)}%</span>
          </div>
          <div className="h-1.5 rounded bg-surface overflow-hidden">
            <div className="h-1.5 rounded bg-brand/60" style={{ width: `${100 * c.value}%` }} />
          </div>
          <div className="text-[10px] text-ink-mute">{c.detail}</div>
        </div>
      ))}
    </div>
  );
}

export default function DesignLab() {
  const ctx = useComposition();
  const [app, setApp] = useState("superconductor");
  const [mustHave, setMustHave] = useState("");
  const [practical, setPractical] = useState(true);

  const apps = useQuery({ queryKey: ["design-apps"], queryFn: api.designApplications });
  const scored = useQuery({
    queryKey: ["design-score", ctx.formula],
    queryFn: () => api.designScore(ctx.formula),
    enabled: !!ctx.formula,
    retry: 0,
  });
  const cands = useQuery({
    queryKey: ["design-candidates", app, mustHave, practical],
    queryFn: () => api.designCandidates(app, 30, mustHave, practical ? RADIOACTIVE : ""),
    retry: 0,
  });

  const list = apps.data?.applications || [];
  const current = list.find((a) => a.id === app);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="font-display text-4xl font-bold">Design Lab</h1>
        <p className="text-ink-soft">
          Pick what you want to build, score your own composition for it, and let the models
          suggest candidates from 554,054 compositions nobody has measured yet.
        </p>
      </div>

      {apps.isLoading ? (
        <Loading text="Loading applications…" />
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-4">
          {list.map((a) => (
            <button
              key={a.id}
              onClick={() => setApp(a.id)}
              className={`card p-5 text-left transition ${
                app === a.id ? "ring-2 ring-brand" : "hover:border-brand/40"
              }`}
            >
              <div className="font-display text-xl font-bold">{a.title}</div>
              <div className="text-xs text-ink-mute">{a.subtitle}</div>
              <p className="mt-2 text-sm text-ink-soft min-h-[40px]">{a.question}</p>
              <div className="mt-2 text-xs">
                <b className="text-brand">{a.n_strong_candidates.toLocaleString()}</b>{" "}
                <span className="text-ink-mute">strong candidates in the pool</span>
              </div>
            </button>
          ))}
        </div>
      )}

      <div className="grid grid-cols-1 xl:grid-cols-[minmax(0,1fr)_minmax(0,1.3fr)] gap-6">
        <div className="card p-5">
          <SectionHeader
            title="Score my composition"
            subtitle={`Current composition: ${ctx.formula}. Change it in the Composition Studio.`}
            right={
              <Link className="btn-ghost text-sm" to="/studio">
                <Search size={15} /> Studio
              </Link>
            }
          />
          {scored.isLoading ? (
            <Loading />
          ) : scored.error ? (
            <ErrorBox error={scored.error} />
          ) : scored.data ? (
            <div className="space-y-4">
              <div className="grid grid-cols-2 sm:grid-cols-5 gap-2 text-center">
                {[
                  ["Stable", scored.data.properties.stable],
                  ["Metal", scored.data.properties.metal],
                  ["Magnetic", scored.data.properties.magnetic],
                  ["PV gap", scored.data.properties.pv_gap],
                ].map(([k, v]) => (
                  <div key={k} className="rounded-lg bg-surface py-2">
                    <div className="label">{k}</div>
                    <div className="font-display text-lg font-bold">
                      {v == null ? "–" : `${(100 * v).toFixed(0)}%`}
                    </div>
                  </div>
                ))}
                <div className="rounded-lg bg-surface py-2">
                  <div className="label">Tc</div>
                  <div className="font-display text-lg font-bold">
                    {scored.data.properties.tc_kelvin == null
                      ? "–"
                      : `${fmt(scored.data.properties.tc_kelvin, 0)} K`}
                  </div>
                </div>
              </div>
              {scored.data.applications.map((r) => (
                <div
                  key={r.application}
                  className={`rounded-xl border p-3 ${
                    r.application === app ? "border-brand bg-brand-50/40" : "border-line"
                  }`}
                >
                  <ScoreBar value={r.score}>
                    <b>{r.title}</b>{" "}
                    <span className="text-ink-mute">· {r.headline} · {r.verdict}</span>
                  </ScoreBar>
                  {r.application === app && <Breakdown components={r.components} />}
                  {r.flags?.length > 0 && (
                    <ul className="mt-2 text-[11px] text-amber-700 list-disc list-inside">
                      {r.flags.map((f) => (
                        <li key={f}>{f}</li>
                      ))}
                    </ul>
                  )}
                </div>
              ))}
            </div>
          ) : null}
        </div>

        <div className="card p-5">
          <SectionHeader
            title={`Suggested ${current?.title.toLowerCase() || ""} candidates`}
            subtitle="Ranked from the unlabeled GNoME pool — compositions proposed by a generative search and never measured."
            right={
              <div className="flex items-center gap-2">
                <label className="flex items-center gap-1.5 text-xs text-ink-soft cursor-pointer">
                  <input type="checkbox" checked={practical} className="accent-indigo-600"
                    onChange={(e) => setPractical(e.target.checked)} />
                  practical elements only
                </label>
                <input
                  className="rounded-xl border border-line bg-surface px-3 py-1.5 text-sm font-mono w-36"
                  placeholder="must contain: Fe,O"
                  value={mustHave}
                  onChange={(e) => setMustHave(e.target.value)}
                />
              </div>
            }
          />
          {current?.note && (
            <p className="mb-3 flex gap-2 rounded-xl bg-surface px-3 py-2 text-xs text-ink-soft">
              <Lightbulb size={14} className="mt-0.5 shrink-0 text-amber-500" />
              {current.note}
            </p>
          )}
          {cands.isLoading ? (
            <Loading text="Ranking candidates…" />
          ) : cands.error ? (
            <ErrorBox error={cands.error} />
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-left text-xs text-ink-mute border-b border-line">
                    <th className="py-2 w-8">#</th>
                    <th>Composition</th>
                    <th>Score</th>
                    <th>Why</th>
                    <th />
                  </tr>
                </thead>
                <tbody>
                  {(cands.data?.candidates || []).map((c, i) => (
                    <tr key={c.formula} className="border-b border-line/60">
                      <td className="py-1.5 text-ink-mute">{i + 1}</td>
                      <td className="font-mono font-semibold">{c.formula}</td>
                      <td>
                        <div className="flex items-center gap-2">
                          <div className="h-2 w-16 rounded bg-surface overflow-hidden">
                            <div className={`h-2 ${BAR(c.score)}`} style={{ width: `${100 * c.score}%` }} />
                          </div>
                          <span className="font-mono text-xs">{(100 * c.score).toFixed(0)}</span>
                        </div>
                      </td>
                      <td className="text-xs text-ink-soft">{c.headline}</td>
                      <td className="text-right">
                        <Link
                          className="text-brand text-xs underline"
                          to={`/results?formula=${encodeURIComponent(c.formula)}`}
                        >
                          open
                        </Link>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {cands.data?.candidates?.length === 0 && (
                <p className="py-6 text-center text-sm text-ink-mute">
                  No candidate matches that filter.
                </p>
              )}
            </div>
          )}
          <p className="mt-3 flex gap-2 text-[11px] text-ink-mute">
            <Sparkles size={13} className="mt-0.5 shrink-0" />
            Screening scores, not a guarantee: every candidate still needs a DFT calculation and
            an experiment.
          </p>
        </div>
      </div>
    </div>
  );
}
