import { FormEvent, useEffect, useMemo, useState } from 'react'

import './App.css'

type Minute = {
  id: string
  title: string
  summary: string
  decisions: string[]
  action_items: string[]
}

type MinuteDraft = {
  title: string
  summary: string
  decisions: string
  action_items: string
}

const API_BASE = import.meta.env.VITE_API_BASE ?? 'http://localhost:8000'

function splitLines(value: string) {
  return value
    .split('\n')
    .map((line) => line.trim())
    .filter(Boolean)
}

function App() {
  const [minutes, setMinutes] = useState<Minute[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [draft, setDraft] = useState<MinuteDraft>({
    title: '',
    summary: '',
    decisions: '',
    action_items: ''
  })

  const isValid = useMemo(
    () => draft.title.trim().length > 0 && draft.summary.trim().length > 0,
    [draft.summary, draft.title]
  )

  useEffect(() => {
    async function loadMinutes() {
      try {
        const response = await fetch(`${API_BASE}/api/minutes`)
        if (!response.ok) {
          throw new Error(`Request failed with status ${response.status}`)
        }
        const payload: Minute[] = await response.json()
        setMinutes(payload)
      } catch (err) {
        console.error(err)
        setError('Unable to load minutes from the API')
      } finally {
        setLoading(false)
      }
    }

    loadMinutes()
  }, [])

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    if (!isValid) return

    const body = {
      title: draft.title.trim(),
      summary: draft.summary.trim(),
      decisions: splitLines(draft.decisions),
      action_items: splitLines(draft.action_items)
    }

    try {
      const response = await fetch(`${API_BASE}/api/minutes`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(body)
      })

      if (!response.ok) {
        throw new Error(`Unable to create minutes (${response.status})`)
      }

      const created: Minute = await response.json()
      setMinutes((current) => [created, ...current])
      setDraft({ title: '', summary: '', decisions: '', action_items: '' })
      setError(null)
    } catch (err) {
      console.error(err)
      setError('Unable to save your minutes. Please try again.')
    }
  }

  return (
    <div className="app">
      <header className="hero">
        <div>
          <p className="eyebrow">Minute Maker</p>
          <h1>Capture decisions and action items fast.</h1>
          <p className="lede">
            A lightweight FastAPI + React starter that keeps meeting highlights organized.
          </p>
          <p className="note">API base: {API_BASE}</p>
        </div>
      </header>

      <main className="content">
        <section className="card">
          <div className="card__header">
            <div>
              <p className="eyebrow">New minutes</p>
              <h2>Create an entry</h2>
            </div>
            <span className="pill">POST /api/minutes</span>
          </div>
          <form className="form" onSubmit={handleSubmit}>
            <label className="field">
              <span>Title</span>
              <input
                name="title"
                placeholder="Sprint retro"
                value={draft.title}
                onChange={(event) =>
                  setDraft((current) => ({ ...current, title: event.target.value }))
                }
                required
              />
            </label>

            <label className="field">
              <span>Summary</span>
              <textarea
                name="summary"
                placeholder="Discussed release schedule and prioritized bug fixes."
                value={draft.summary}
                onChange={(event) =>
                  setDraft((current) => ({ ...current, summary: event.target.value }))
                }
                required
                rows={3}
              />
            </label>

            <label className="field">
              <span>Decisions (one per line)</span>
              <textarea
                name="decisions"
                placeholder="Extend beta by one week\nShip hotfix for onboarding"
                value={draft.decisions}
                onChange={(event) =>
                  setDraft((current) => ({ ...current, decisions: event.target.value }))
                }
                rows={3}
              />
            </label>

            <label className="field">
              <span>Action items (one per line)</span>
              <textarea
                name="action_items"
                placeholder="Update release notes\nSchedule customer calls"
                value={draft.action_items}
                onChange={(event) =>
                  setDraft((current) => ({ ...current, action_items: event.target.value }))
                }
                rows={3}
              />
            </label>

            <div className="form__footer">
              <button className="primary" type="submit" disabled={!isValid}>
                Save minutes
              </button>
              {error && <p className="error">{error}</p>}
            </div>
          </form>
        </section>

        <section className="card">
          <div className="card__header">
            <div>
              <p className="eyebrow">Minutes</p>
              <h2>Recent notes</h2>
            </div>
            <span className="pill">GET /api/minutes</span>
          </div>
          {loading ? (
            <p className="muted">Loading minutes…</p>
          ) : minutes.length === 0 ? (
            <p className="muted">No minutes yet. Add the first entry above.</p>
          ) : (
            <ul className="minute-list">
              {minutes.map((minute) => (
                <li className="minute" key={minute.id}>
                  <div className="minute__header">
                    <div>
                      <p className="eyebrow">{minute.id}</p>
                      <h3>{minute.title}</h3>
                    </div>
                    <span className="pill pill--secondary">{minute.decisions.length} decisions</span>
                  </div>
                  <p className="summary">{minute.summary}</p>

                  {minute.decisions.length > 0 && (
                    <div className="stack">
                      <p className="eyebrow">Decisions</p>
                      <ul className="bullets">
                        {minute.decisions.map((decision, index) => (
                          <li key={`${minute.id}-decision-${index}`}>{decision}</li>
                        ))}
                      </ul>
                    </div>
                  )}

                  {minute.action_items.length > 0 && (
                    <div className="stack">
                      <p className="eyebrow">Action items</p>
                      <ul className="bullets">
                        {minute.action_items.map((item, index) => (
                          <li key={`${minute.id}-action-${index}`}>{item}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                </li>
              ))}
            </ul>
          )}
        </section>
      </main>
    </div>
  )
}

export default App
