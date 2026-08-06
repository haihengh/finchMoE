'use client'

import { useState, useMemo } from 'react'
import { motion } from 'framer-motion'
import { Plus, Minus, RotateCcw } from 'lucide-react'
import { MODEL_CONFIG, kvCacheMB } from '@/lib/data'

export default function MemoryTab() {
  const [seqLen, setSeqLen] = useState(512)
  const [extraExperts, setExtraExperts] = useState(0)
  const [evictionFlash, setEvictionFlash] = useState(false)

  const budgetGB = MODEL_CONFIG.cache_budget_gb
  const denseMB = MODEL_CONFIG.dense_backbone_mb
  const activationMB = MODEL_CONFIG.activation_scratch_mb
  const kvMB = kvCacheMB(seqLen)
  const expertMB = extraExperts * MODEL_CONFIG.expert_slab_mb
  const totalMB = denseMB + kvMB + activationMB + expertMB
  const totalGB = totalMB / 1024
  const pct = Math.min((totalGB / budgetGB) * 100, 100)
  const headroomMB = Math.max(0, budgetGB * 1024 - denseMB - kvMB - activationMB)
  const maxExperts = Math.floor(headroomMB / MODEL_CONFIG.expert_slab_mb)

  const gaugeColor = pct < 87.5 ? '#10b981' : pct < 95 ? '#f59e0b' : '#ef4444'

  const addExperts = (n: number) => {
    const newCount = Math.max(0, extraExperts + n)
    const newTotal = denseMB + kvMB + activationMB + newCount * MODEL_CONFIG.expert_slab_mb
    if (newTotal > budgetGB * 1024) {
      setEvictionFlash(true)
      setTimeout(() => setEvictionFlash(false), 600)
    }
    setExtraExperts(Math.min(newCount, maxExperts + 5))
  }

  const segments = useMemo(() => [
    { label: 'Dense Backbone', mb: denseMB, color: '#3b82f6' },
    { label: 'KV Cache', mb: kvMB, color: '#8b5cf6' },
    { label: 'Activations', mb: activationMB, color: '#a855f7' },
    { label: 'Expert Cache', mb: Math.min(expertMB, headroomMB), color: '#06b6d4' },
  ], [denseMB, kvMB, activationMB, expertMB, headroomMB])

  // SVG gauge parameters
  const radius = 100
  const stroke = 14
  const circumference = Math.PI * radius
  const dashOffset = circumference - (pct / 100) * circumference

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl lg:text-3xl font-display font-bold tracking-tight">Memory Manager</h2>
        <p className="text-muted-foreground mt-1">Live memory budget simulation for {budgetGB} GB RAM cap</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Gauge */}
        <div className="glass-card rounded-xl p-6 border border-border/50 flex flex-col items-center">
          <div className="relative" style={{ width: 240, height: 140 }}>
            <svg width="240" height="140" viewBox="0 0 240 140">
              {/* Background arc */}
              <path
                d="M 20 130 A 100 100 0 0 1 220 130"
                fill="none"
                stroke="hsl(var(--muted) / 0.3)"
                strokeWidth={stroke}
                strokeLinecap="round"
              />
              {/* Filled arc */}
              <motion.path
                d="M 20 130 A 100 100 0 0 1 220 130"
                fill="none"
                stroke={gaugeColor}
                strokeWidth={stroke}
                strokeLinecap="round"
                strokeDasharray={circumference}
                animate={{ strokeDashoffset: dashOffset }}
                transition={{ duration: 0.5 }}
                style={{ filter: `drop-shadow(0 0 8px ${gaugeColor}40)` }}
              />
            </svg>
            <div className="absolute inset-0 flex flex-col items-center justify-end pb-2">
              <span className="text-3xl font-display font-bold" style={{ color: gaugeColor }}>
                {totalGB.toFixed(2)}
              </span>
              <span className="text-xs text-muted-foreground">/ {budgetGB}.00 GB</span>
            </div>
          </div>

          {/* Zone labels */}
          <div className="flex items-center gap-4 mt-4 text-xs">
            <span className="flex items-center gap-1">
              <div className="w-2 h-2 rounded-full bg-green-500" /> 0–3.5 GB
            </span>
            <span className="flex items-center gap-1">
              <div className="w-2 h-2 rounded-full bg-yellow-500" /> 3.5–3.8 GB
            </span>
            <span className="flex items-center gap-1">
              <div className="w-2 h-2 rounded-full bg-red-500" /> 3.8–4.0 GB
            </span>
          </div>
        </div>

        {/* Controls */}
        <div className="glass-card rounded-xl p-6 border border-border/50 space-y-6">
          {/* Context length slider */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <label className="text-sm font-medium">Context Length</label>
              <span className="text-sm font-mono text-primary">{seqLen} tokens</span>
            </div>
            <input
              type="range"
              min={0}
              max={4096}
              step={64}
              value={seqLen}
              onChange={(e: React.ChangeEvent<HTMLInputElement>) => setSeqLen(Number(e?.target?.value ?? 0))}
              className="w-full accent-primary"
            />
            <p className="text-xs text-muted-foreground mt-1">
              KV Cache: {kvMB.toFixed(1)} MB ({MODEL_CONFIG.num_hidden_layers} layers × {MODEL_CONFIG.num_key_value_heads} KV heads × {MODEL_CONFIG.head_dim} dim × {seqLen} × 4 bytes)
            </p>
          </div>

          {/* Expert loader */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <label className="text-sm font-medium">Expert Slabs Loaded</label>
              <span className="text-sm font-mono text-accent">{extraExperts} × {MODEL_CONFIG.expert_slab_mb} MB</span>
            </div>
            <div className="flex items-center gap-2">
              <button
                onClick={() => addExperts(-10)}
                className="p-2 rounded-lg bg-muted/50 hover:bg-muted text-muted-foreground transition-colors"
              >
                <Minus className="w-4 h-4" />
              </button>
              <button
                onClick={() => addExperts(10)}
                className="flex items-center gap-1 px-4 py-2 rounded-lg bg-accent/20 hover:bg-accent/30 text-accent font-medium text-sm transition-colors"
              >
                <Plus className="w-4 h-4" /> Load +10
              </button>
              <button
                onClick={() => { setExtraExperts(0); setSeqLen(512) }}
                className="p-2 rounded-lg bg-muted/50 hover:bg-muted text-muted-foreground transition-colors"
              >
                <RotateCcw className="w-4 h-4" />
              </button>
              <span className="text-xs text-muted-foreground ml-auto">
                Max: ~{maxExperts} slabs
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Breakdown bars */}
      <div className={`glass-card rounded-xl p-6 border transition-all duration-300 ${
        evictionFlash ? 'border-destructive/60 glow-blue' : 'border-border/50'
      }`}>
        <p className="text-xs font-mono text-muted-foreground uppercase tracking-wider mb-4">Memory Breakdown</p>
        <div className="space-y-3">
          {segments.map((seg: any) => {
            const segPct = ((seg?.mb ?? 0) / (budgetGB * 1024)) * 100
            return (
              <div key={seg?.label} className="space-y-1">
                <div className="flex items-center justify-between text-sm">
                  <span className="flex items-center gap-2">
                    <div className="w-3 h-3 rounded" style={{ backgroundColor: seg?.color ?? '#3b82f6' }} />
                    {seg?.label}
                  </span>
                  <span className="font-mono text-muted-foreground">
                    {(seg?.mb ?? 0).toFixed(1)} MB ({segPct.toFixed(1)}%)
                  </span>
                </div>
                <div className="w-full h-3 rounded-full bg-muted/20 overflow-hidden">
                  <motion.div
                    className="h-full rounded-full"
                    style={{ backgroundColor: seg?.color ?? '#3b82f6' }}
                    animate={{ width: `${Math.min(segPct, 100)}%` }}
                    transition={{ duration: 0.4 }}
                  />
                </div>
              </div>
            )
          })}
          {/* Remaining headroom */}
          <div className="space-y-1">
            <div className="flex items-center justify-between text-sm">
              <span className="flex items-center gap-2">
                <div className="w-3 h-3 rounded bg-muted-foreground/20" />
                Free Headroom
              </span>
              <span className="font-mono text-muted-foreground">
                {Math.max(0, budgetGB * 1024 - totalMB).toFixed(1)} MB
              </span>
            </div>
          </div>
        </div>

        {totalGB > budgetGB && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="mt-4 p-3 rounded-lg bg-destructive/10 border border-destructive/30 text-sm text-destructive"
          >
            ⚠️ Budget exceeded! Hard eviction triggered. Remove expert slabs or reduce context length.
          </motion.div>
        )}
      </div>
    </div>
  )
}
