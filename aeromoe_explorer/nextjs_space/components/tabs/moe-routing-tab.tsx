'use client'

import { useState, useEffect, useRef, useCallback } from 'react'
import { motion } from 'framer-motion'
import { Shuffle, Play, Pause, HardDrive, Cpu as CpuIcon } from 'lucide-react'
import { MODEL_CONFIG } from '@/lib/data'

type ExpertStatus = 'cold' | 'loading' | 'hot'

interface ExpertState {
  id: number
  score: number
  selected: boolean
  cacheStatus: ExpertStatus
  frequency: number
}

function initExperts(): ExpertState[] {
  return Array.from({ length: MODEL_CONFIG.num_experts }, (_: unknown, i: number) => ({
    id: i,
    score: 0,
    selected: false,
    cacheStatus: i < 40 ? 'hot' : 'cold' as ExpertStatus,
    frequency: i < 40 ? 10 - Math.floor(i / 5) : 0,
  }))
}

function pickTop8(experts: ExpertState[]): { indices: number[]; scores: number[] } {
  const indices: number[] = []
  const scores: number[] = []
  const used = new Set<number>()
  for (let k = 0; k < 8; k++) {
    let idx = Math.floor(Math.random() * 128)
    while (used.has(idx)) idx = Math.floor(Math.random() * 128)
    used.add(idx)
    indices.push(idx)
    scores.push(0.5 + Math.random() * 0.5)
  }
  // sort by score desc
  const sorted = indices.map((idx: number, i: number) => ({ idx, score: scores?.[i] ?? 0 }))
    .sort((a: any, b: any) => (b?.score ?? 0) - (a?.score ?? 0))
  return {
    indices: sorted.map((s: any) => s?.idx ?? 0),
    scores: sorted.map((s: any) => s?.score ?? 0),
  }
}

export default function MoeRoutingTab() {
  const [experts, setExperts] = useState<ExpertState[]>(initExperts)
  const [cacheUsed, setCacheUsed] = useState(40)
  const [autoMode, setAutoMode] = useState(false)
  const [evictionLog, setEvictionLog] = useState<string[]>([])
  const autoRef = useRef<NodeJS.Timeout | null>(null)

  const generateRouting = useCallback(() => {
    const { indices, scores } = pickTop8(experts)
    setExperts((prev: ExpertState[]) => {
      const next = (prev ?? []).map((e: ExpertState) => ({
        ...(e ?? {}),
        selected: false,
        score: 0,
      }))
      let loaded = next?.filter?.((e: ExpertState) => e?.cacheStatus === 'hot')?.length ?? 0

      indices.forEach((idx: number, i: number) => {
        const expert = next?.[idx]
        if (!expert) return
        expert.selected = true
        expert.score = scores?.[i] ?? 0
        expert.frequency = (expert.frequency ?? 0) + 1

        if (expert.cacheStatus === 'cold') {
          if (loaded >= MODEL_CONFIG.cache_capacity_slabs) {
            // Evict LFU
            const coldCandidates = next
              .filter((e: ExpertState) => e?.cacheStatus === 'hot' && !e?.selected)
              .sort((a: ExpertState, b: ExpertState) => (a?.frequency ?? 0) - (b?.frequency ?? 0))
            if (coldCandidates?.[0]) {
              const victim = coldCandidates[0]
              victim.cacheStatus = 'cold'
              loaded--
              setEvictionLog((p: string[]) => [
                `Evicted E${victim?.id} (freq=${victim?.frequency})`,
                ...((p ?? []).slice(0, 9)),
              ])
            }
          }
          expert.cacheStatus = 'loading'
          loaded++
          setTimeout(() => {
            setExperts((p: ExpertState[]) => {
              const copy = [...(p ?? [])]
              if (copy?.[idx]) copy[idx] = { ...(copy[idx] ?? {}), cacheStatus: 'hot' }
              return copy
            })
          }, 800)
        }
      })

      setCacheUsed(next?.filter?.((e: ExpertState) => e?.cacheStatus === 'hot' || e?.cacheStatus === 'loading')?.length ?? 0)
      return next
    })
  }, [experts])

  useEffect(() => {
    if (autoMode) {
      autoRef.current = setInterval(generateRouting, 2000)
    }
    return () => {
      if (autoRef?.current) clearInterval(autoRef.current)
    }
  }, [autoMode, generateRouting])

  const memoryUsed = cacheUsed * MODEL_CONFIG.expert_slab_mb
  const memoryTotal = MODEL_CONFIG.cache_capacity_slabs * MODEL_CONFIG.expert_slab_mb

  const selectedExperts = (experts ?? []).filter((e: ExpertState) => e?.selected).sort((a: ExpertState, b: ExpertState) => (b?.score ?? 0) - (a?.score ?? 0))

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h2 className="text-2xl lg:text-3xl font-display font-bold tracking-tight">MoE Expert Routing</h2>
          <p className="text-muted-foreground mt-1">Interactive Mixture-of-Experts routing simulation</p>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={generateRouting}
            className="flex items-center gap-2 px-4 py-2 rounded-lg bg-primary text-primary-foreground font-medium text-sm hover:bg-primary/90 transition-colors glow-blue"
          >
            <Shuffle className="w-4 h-4" />
            New Token
          </button>
          <button
            onClick={() => setAutoMode(!autoMode)}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium text-sm transition-colors ${
              autoMode ? 'bg-accent text-accent-foreground glow-cyan' : 'bg-muted/50 text-muted-foreground hover:bg-muted'
            }`}
          >
            {autoMode ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
            Auto
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Expert grid */}
        <div className="lg:col-span-2 glass-card rounded-xl p-5 border border-border/50">
          <p className="text-xs font-mono text-muted-foreground uppercase tracking-wider mb-3">
            128 Experts (16×8 Grid)
          </p>
          <div className="grid grid-cols-16 gap-1">
            {(experts ?? []).map((e: ExpertState) => {
              const isSelected = e?.selected
              const statusColor = e?.cacheStatus === 'hot'
                ? 'bg-green-500/20 border-green-500/40'
                : e?.cacheStatus === 'loading'
                ? 'bg-yellow-500/20 border-yellow-500/40'
                : 'bg-muted/20 border-border/30'
              return (
                <motion.div
                  key={e?.id}
                  className={`aspect-square rounded-sm border text-center flex items-center justify-center transition-all duration-300 ${
                    isSelected
                      ? 'bg-primary border-primary/60 glow-blue z-10 scale-110'
                      : statusColor
                  }`}
                  animate={isSelected ? { scale: [1, 1.2, 1.1] } : { scale: 1 }}
                  transition={{ duration: 0.3 }}
                  title={`E${e?.id} | ${e?.cacheStatus} | freq: ${e?.frequency}`}
                >
                  <span className={`text-[7px] font-mono leading-none ${
                    isSelected ? 'text-primary-foreground' : 'text-muted-foreground/50'
                  }`}>
                    {e?.id}
                  </span>
                </motion.div>
              )
            })}
          </div>

          {/* Legend */}
          <div className="flex items-center gap-4 mt-3">
            <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
              <div className="w-3 h-3 rounded-sm bg-primary" /> Selected (top-8)
            </div>
            <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
              <div className="w-3 h-3 rounded-sm bg-green-500/30 border border-green-500/50" /> Hot (cached)
            </div>
            <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
              <div className="w-3 h-3 rounded-sm bg-yellow-500/30 border border-yellow-500/50" /> Loading
            </div>
            <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
              <div className="w-3 h-3 rounded-sm bg-muted/20 border border-border/30" /> Cold
            </div>
          </div>
        </div>

        {/* Right panel */}
        <div className="space-y-4">
          {/* Top-8 winners */}
          <div className="glass-card rounded-xl p-4 border border-border/50">
            <p className="text-xs font-mono text-muted-foreground uppercase tracking-wider mb-3">Top-8 Selected</p>
            {selectedExperts?.length > 0 ? (
              <div className="space-y-1.5">
                {selectedExperts.map((e: ExpertState, i: number) => (
                  <div key={e?.id} className="flex items-center gap-2">
                    <span className="text-xs font-mono text-primary w-8">E{e?.id}</span>
                    <div className="flex-1 h-2 rounded-full bg-muted/30 overflow-hidden">
                      <motion.div
                        className="h-full rounded-full bg-gradient-to-r from-primary to-accent"
                        initial={{ width: 0 }}
                        animate={{ width: `${(e?.score ?? 0) * 100}%` }}
                        transition={{ duration: 0.4 }}
                        style={{ opacity: 1 - i * 0.08 }}
                      />
                    </div>
                    <span className="text-xs font-mono text-muted-foreground w-10 text-right">
                      {(e?.score ?? 0).toFixed(2)}
                    </span>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-xs text-muted-foreground">Click "New Token" to generate routing</p>
            )}
          </div>

          {/* Cache status */}
          <div className="glass-card rounded-xl p-4 border border-border/50">
            <p className="text-xs font-mono text-muted-foreground uppercase tracking-wider mb-3">
              <HardDrive className="w-3 h-3 inline mr-1" />
              Expert Cache
            </p>
            <div className="space-y-2">
              <div className="flex justify-between text-xs">
                <span className="text-muted-foreground">Slabs loaded</span>
                <span className="font-mono">{cacheUsed}/{MODEL_CONFIG.cache_capacity_slabs}</span>
              </div>
              <div className="w-full h-2 rounded-full bg-muted/30 overflow-hidden">
                <div
                  className="h-full rounded-full bg-gradient-to-r from-accent to-primary transition-all duration-500"
                  style={{ width: `${(cacheUsed / MODEL_CONFIG.cache_capacity_slabs) * 100}%` }}
                />
              </div>
              <div className="flex justify-between text-xs">
                <span className="text-muted-foreground">Memory</span>
                <span className="font-mono">
                  {(memoryUsed / 1024).toFixed(2)} GB / {(memoryTotal / 1024).toFixed(2)} GB
                </span>
              </div>
            </div>
          </div>

          {/* Eviction log */}
          <div className="glass-card rounded-xl p-4 border border-border/50">
            <p className="text-xs font-mono text-muted-foreground uppercase tracking-wider mb-3">
              Eviction Log
            </p>
            <div className="space-y-1 max-h-32 overflow-y-auto scrollbar-none">
              {(evictionLog?.length ?? 0) > 0 ? (
                evictionLog.map((log: string, i: number) => (
                  <motion.p
                    key={`${log}-${i}`}
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1 - i * 0.1, x: 0 }}
                    className="text-xs font-mono text-destructive/80"
                  >
                    {log}
                  </motion.p>
                ))
              ) : (
                <p className="text-xs text-muted-foreground">No evictions yet</p>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
