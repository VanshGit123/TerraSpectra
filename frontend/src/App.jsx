import React, { useCallback, useEffect, useState } from 'react'
import { Area, AreaChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import RiskMap from './components/RiskMap'

const API = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000'
const demoTrend = [
  { date: 'Week 1', health: 94, stress: 6 }, { date: 'Week 2', health: 92, stress: 8 },
  { date: 'Week 3', health: 89, stress: 11 }, { date: 'Week 4', health: 86, stress: 14 },
  { date: 'Week 5', health: 82, stress: 18 }, { date: 'Current', health: 79, stress: 21 },
]

export default function App() {
  const [metrics, setMetrics] = useState(null)
  const [farm, setFarm] = useState(null)
  const [geojson, setGeojson] = useState({type:'FeatureCollection',features:[]})
  const [loading, setLoading] = useState(true)
  const [running, setRunning] = useState(false)
  const [error, setError] = useState('')

  const refresh = useCallback(async () => {
    setLoading(true); setError('')
    try {
      const [m, f, l] = await Promise.all([fetch(`${API}/api/metrics`), fetch(`${API}/api/farms`), fetch(`${API}/api/layers`)])
      if (!m.ok || !f.ok || !l.ok) throw new Error('API request failed. Is the FastAPI server running?')
      const [mj, fj, lj] = await Promise.all([m.json(), f.json(), l.json()])
      setMetrics(mj); setFarm(fj.farms[0]); setGeojson(lj)
    } catch (e) { setError(e.message) } finally { setLoading(false) }
  }, [])
  useEffect(() => { refresh() }, [refresh])

  async function runInference() {
    setRunning(true); setError('')
    try {
      const response = await fetch(`${API}/api/predict`, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({tile_size:32,overlap:8})})
      if (!response.ok) throw new Error('Inference failed. Check the backend logs.')
      await refresh()
    } catch (e) { setError(e.message) } finally { setRunning(false) }
  }

  const acres = n => Number(n || 0).toLocaleString('en-IN', {maximumFractionDigits:1})
  return <div className="app-shell">
    <aside className="sidebar"><div className="brand"><div className="brand-mark">✳</div><div><b>TerraSpectra</b><small>FIELD INTELLIGENCE</small></div></div>
      <div className="nav-caption">WORKSPACE</div><a className="nav-item active">▦ <span>Overview</span></a><a className="nav-item">⌖ <span>Field monitoring</span></a><a className="nav-item">◈ <span>Spectral analysis</span></a><a className="nav-item">◷ <span>History & trends</span></a>
      <div className="sidebar-bottom"><div className="status-dot"/> Demo environment <small>v1.0.0</small></div>
    </aside>
    <main className="main-content"><header className="topbar"><div><span className="eyebrow">PRECISION AGRICULTURE / DASHBOARD</span><h1>Crop health overview</h1></div><div className="top-actions"><span className="live-pill">● DEMO DATA</span><button className="run-button" onClick={runInference} disabled={running}>{running ? 'Processing…' : '↻ Run analysis'}</button></div></header>
      {error && <div className="error-banner">{error} <button onClick={refresh}>Retry</button></div>}
      <section className="welcome-row"><div><h2>{farm?.name || 'TerraSpectra Demo Farm'}</h2><p>Hyperspectral crop stress monitoring · {farm?.area_acres || 1000} acres</p></div><div className="date-label">⌖ {farm ? `${farm.center.latitude.toFixed(4)}° N, ${farm.center.longitude.toFixed(4)}° E` : 'Loading location…'}</div></section>
      <section className="metric-grid">
        <Metric title="TOTAL FARM AREA" value={`${acres(metrics?.total_acres)} ac`} note="Monitored field area" icon="▧" />
        <Metric title="HIGH RISK AREA" value={`${acres(metrics?.high_risk_acres)} ac`} note="Stress probability ≥ 70%" icon="△" tone="red" />
        <Metric title="MEDIUM RISK AREA" value={`${acres(metrics?.medium_risk_acres)} ac`} note="Stress probability 40–69%" icon="◒" tone="amber" />
        <Metric title="MEAN STRESS SCORE" value={metrics ? `${(metrics.mean_stress_probability*100).toFixed(1)}%` : '—'} note="Average pixel probability" icon="⌁" tone="green" />
      </section>
      <section className="content-grid"><div className="panel map-panel"><div className="panel-heading"><div><h3>Geospatial risk map</h3><p>Pixel-level simulated crop stress probability</p></div><span className="layer-tag">STRESS LAYER · ON</span></div><RiskMap geojson={geojson} center={farm?.center || {latitude:29.9457,longitude:78.1642}}/><div className="map-legend"><span><i className="low"/>Low risk</span><span><i className="medium"/>Medium risk</span><span><i className="high"/>High risk</span></div></div>
        <div className="panel insight-panel"><div className="panel-heading"><div><h3>Field insights</h3><p>Current model assessment</p></div><span className="spark-icon">✧</span></div><div className="insight-score"><div className="score-ring"><b>{metrics ? Math.round(metrics.mean_stress_probability*100) : '—'}</b><small>/ 100</small></div><div><b>Average stress index</b><p>Across the monitored farm</p></div></div><div className="insight-divider"/><div className="insight-line"><span>High-risk pixels</span><b>{acres(metrics?.high_risk_pixels)}</b></div><div className="insight-line"><span>Inference engine</span><b className="engine-name">{metrics?.model_source?.includes('hybrid') ? '3D-CNN + ViT' : 'Spectral heuristic'}</b></div><div className="notice">Model output is a simulated stress indicator, not a confirmed disease diagnosis.</div></div></section>
      <section className="panel trend-panel"><div className="panel-heading"><div><h3>Spectral health timeline</h3><p>Illustrative historical trend · replace with dated observations</p></div><span className="period-tag">6 PERIODS</span></div><div className="chart-wrap"><ResponsiveContainer width="100%" height="100%"><AreaChart data={demoTrend} margin={{top:10,right:16,left:-18,bottom:0}}><defs><linearGradient id="healthFill" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#2c9b79" stopOpacity={.22}/><stop offset="95%" stopColor="#2c9b79" stopOpacity={0}/></linearGradient></defs><CartesianGrid strokeDasharray="3 5" vertical={false} stroke="#edf0ed"/><XAxis dataKey="date" axisLine={false} tickLine={false} tick={{fill:'#8a9690',fontSize:11}}/><YAxis domain={[0,100]} axisLine={false} tickLine={false} tick={{fill:'#8a9690',fontSize:11}}/><Tooltip/><Area type="monotone" dataKey="health" name="Health index" stroke="#278b70" strokeWidth={2.5} fill="url(#healthFill)"/></AreaChart></ResponsiveContainer></div></section>
      <footer>TerraSpectra · Internship MVP · Synthetic demonstration data</footer>
    </main>
  </div>
}

function Metric({title,value,note,icon,tone=''}) { return <div className="metric-card"><div className="metric-top"><span>{title}</span><i className={tone}>{icon}</i></div><strong>{value || '—'}</strong><small>{note}</small></div> }
