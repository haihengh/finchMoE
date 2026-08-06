'use client'

import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { ChevronRight, FileCode2, Zap, X } from 'lucide-react'
import { ARCHITECTURE_LAYERS, SESSION_LABELS } from '@/lib/data'
import type { ArchitectureLayer } from '@/lib/data'

export default function OverviewTab() {
  const [selectedLayer, setSelectedLayer] = useState<ArchitectureLayer | null>(null)
  const [pulseActive, setPulseActive] = useState(false)
  const [pulseIndex, setPulseIndex] = useState(-1)

  const runPulse = () => {
    if (pulseActive) return
    setPulseActive(true)
    setPulseIndex(ARCHITECTURE_LAYERS.length - 1)
    const totalLayers = ARCHITECTURE_LAYERS?.length ?? 0
    let idx = totalLayers - 1
    const interval = setInterval(() => {
      idx--
      if (idx < 0) {
        clearInterval(interval)
        setPulseActive(false)
        setPulseIndex(-1)
        return
      }
      setPulseIndex(idx)
    }, 400)
  }

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h2 className="text-2xl lg:text-3xl font-display font-bold tracking-tight">Architecture Stack</h2>
          <p className="text-muted-foreground mt-1">Click a layer to inspect its components, or run a signal pulse.</p>
        </div>
        <button
          onClick={runPulse}
          disabled={pulseActive}
          className="flex items-center gap-2 px-4 py-2 rounded-lg bg-primary text-primary-foreground font-medium text-sm hover:bg-primary/90 transition-colors disabled:opacity-50 glow-blue"
        >
          <Zap className="w-4 h-4" />
          Run Inference
        </button>
      </div>

      <div className="flex flex-col lg:flex-row gap-6">
        {/* Stack diagram */}
        <div className="flex-1 space-y-2">
          {ARCHITECTURE_LAYERS?.map?.((layer: ArchitectureLayer, i: number) => {
            const isPulsing = pulseIndex === i
            const isSelected = selectedLayer?.id === layer?.id
            const sessionInfo = SESSION_LABELS?.[layer?.session ?? 0]
            return (
              <motion.button
                key={layer?.id}
                onClick={() => setSelectedLayer(isSelected ? null : layer)}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: i * 0.08 }}
                className={`w-full text-left p-4 rounded-xl border transition-all duration-300 ${
                  isSelected
                    ? 'border-primary/50 bg-primary/10 glow-blue'
                    : isPulsing
                    ? 'border-accent/50 bg-accent/10 glow-cyan'
                    : 'border-border/40 glass-card hover:border-primary/30 hover:bg-primary/5'
                }`}
                style={isPulsing ? { boxShadow: `0 0 30px ${layer?.color ?? '#3b82f6'}40` } : {}}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div
                      className="w-3 h-3 rounded-full"
                      style={{ backgroundColor: layer?.color ?? '#3b82f6' }}
                    />
                    <span className="font-display font-semibold text-sm lg:text-base">{layer?.label}</span>
                    {sessionInfo && layer?.session !== 0 && (
                      <span className="text-xs px-2 py-0.5 rounded-full bg-muted/50 text-muted-foreground font-mono">
                        Session {layer?.session}
                      </span>
                    )}
                  </div>
                  <ChevronRight className={`w-4 h-4 text-muted-foreground transition-transform ${isSelected ? 'rotate-90' : ''}`} />
                </div>
                <div className="flex flex-wrap gap-2 mt-2">
                  {layer?.components?.map?.((comp: string, j: number) => (
                    <span
                      key={j}
                      className="text-xs px-2 py-1 rounded-md font-mono border border-border/50"
                      style={{ color: layer?.color ?? '#3b82f6', borderColor: `${layer?.color ?? '#3b82f6'}30` }}
                    >
                      {comp}
                    </span>
                  )) ?? []}
                </div>
              </motion.button>
            )
          }) ?? []}

          {/* Animated arrows between layers */}
          <div className="flex justify-center py-1">
            <div className="flex flex-col items-center gap-0.5">
              {[0, 1, 2].map((i: number) => (
                <motion.div
                  key={i}
                  className="w-1 h-1 rounded-full bg-primary/40"
                  animate={{ opacity: [0.3, 1, 0.3] }}
                  transition={{ duration: 1.5, repeat: Infinity, delay: i * 0.3 }}
                />
              ))}
            </div>
          </div>
        </div>

        {/* Detail panel */}
        <AnimatePresence>
          {selectedLayer && (
            <motion.div
              initial={{ opacity: 0, x: 20, width: 0 }}
              animate={{ opacity: 1, x: 0, width: 'auto' }}
              exit={{ opacity: 0, x: 20, width: 0 }}
              className="w-full lg:w-80 xl:w-96"
            >
              <div className="glass-card rounded-xl p-5 border border-border/50 sticky top-24">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="font-display font-bold text-lg" style={{ color: selectedLayer?.color ?? '#3b82f6' }}>
                    {selectedLayer?.label}
                  </h3>
                  <button onClick={() => setSelectedLayer(null)} className="p-1 rounded hover:bg-muted/50">
                    <X className="w-4 h-4 text-muted-foreground" />
                  </button>
                </div>
                <p className="text-sm text-muted-foreground mb-4">{selectedLayer?.description}</p>

                <div className="space-y-3">
                  <div>
                    <p className="text-xs font-mono text-muted-foreground uppercase tracking-wider mb-2">Components</p>
                    <div className="flex flex-wrap gap-1.5">
                      {selectedLayer?.components?.map?.((c: string, i: number) => (
                        <span key={i} className="text-xs px-2 py-1 rounded-md bg-muted/50 text-foreground font-mono">
                          {c}
                        </span>
                      )) ?? []}
                    </div>
                  </div>

                  {(selectedLayer?.files?.length ?? 0) > 0 && (
                    <div>
                      <p className="text-xs font-mono text-muted-foreground uppercase tracking-wider mb-2">Source Files</p>
                      <div className="space-y-1">
                        {selectedLayer?.files?.map?.((f: string, i: number) => (
                          <div key={i} className="flex items-center gap-2 text-sm text-muted-foreground">
                            <FileCode2 className="w-3 h-3 flex-shrink-0" />
                            <span className="font-mono text-xs">{f}</span>
                          </div>
                        )) ?? []}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  )
}
