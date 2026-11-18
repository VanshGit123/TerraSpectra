import React, { useMemo } from 'react'
import DeckGL from '@deck.gl/react'
import { GeoJsonLayer } from '@deck.gl/layers'
import Map from 'react-map-gl/mapbox'
import 'mapbox-gl/dist/mapbox-gl.css'

const token = import.meta.env.VITE_MAPBOX_TOKEN
const colorFor = value => value >= .7 ? [220, 65, 70, 175] : value >= .4 ? [245, 174, 65, 165] : [55, 166, 125, 135]

export default function RiskMap({ geojson, center }) {
  const layers = useMemo(() => [new GeoJsonLayer({
    id: 'crop-risk-layer', data: geojson, filled: true, stroked: false,
    getFillColor: f => colorFor(f.properties.risk), pickable: true,
    autoHighlight: true, highlightColor: [255,255,255,100]
  })], [geojson])
  if (!token) return <div className="map-fallback"><div className="fallback-label">DEMO FIELD GRID · RISK OVERLAY</div><div className="grid-map">{(geojson?.features || []).map((f, i) => <span key={i} title={`Risk ${(f.properties.risk*100).toFixed(0)}%`} style={{background: `rgba(${colorFor(f.properties.risk).slice(0,3).join(',')},.78)`}} />)}</div><p>Add <code>VITE_MAPBOX_TOKEN</code> to enable the interactive geographic basemap.</p></div>
  return <div className="map-wrap"><DeckGL initialViewState={{longitude:center.longitude, latitude:center.latitude, zoom:14, pitch:35, bearing:-12}} controller layers={layers}><Map mapboxAccessToken={token} mapStyle="mapbox://styles/mapbox/satellite-streets-v12" /></DeckGL></div>
}
