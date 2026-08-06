'use client'

import { useState, useEffect, useCallback, useRef } from 'react'
import { motion } from 'framer-motion'
import { Play, Pause, SkipForward, RotateCcw, ChevronRight } from 'lucide-react'
import { PIPELINE_STEPS, MODEL_CONFIG } from '@/lib/data'

const EXAMPLE_PROMPT = 'What is 2+2?'
const EXAMPLE_TOKENS = [
  { text: 'What', id: 3838 },
  { text: ' is', id: 374 },
  { text: ' 2', id: 220 },
  { text: '+', id: 10 },
  { text: '2', id: 17 },
  { text: '?', id: 30 },
]

export default function PipelineTab() {
  const [currentStep, setCurrentStep] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [layerCount, setLayerCount] = useState(0)
  const [typedChars, setTypedChars] = useState(0)
  const timerRef = useRef<NodeJS.Timeout | null>(null)

  const stepCount = PIPELINE_STEPS?.length ?? 0

  const advanceStep = useCallback(() => {
    setCurrentStep((prev: number) => {
      if (prev >= stepCount - 1) {
        setPlaying(false)
        return prev
      }
      return prev + 1
    })
  }, [stepCount])

  useEffect(() => {
    if (playing) {
      timerRef.current = setInterval(advanceStep, 2000)
    }
    return () => {
      if (timerRef?.current) clearInterval(timerRef.current)
    }
  }, [playing, advanceStep])

  // Typing animation for step 0
  useEffect(() => {
    if (currentStep === 0) {
      setTypedChars(0)
      let i = 0
      const iv = setInterval(() => {
        i++
        setTypedChars(i)
        if (i >= (EXAMPLE_PROMPT?.length ?? 0)) clearInterval(iv)
      }, 80)
      return () => clearInterval(iv)
    }
  }, [currentStep])

  // Layer counter for step 4
  useEffect(() => {
    if (currentStep === 4) {
      setLayerCount(0)
      let l = 0
      const iv = setInterval(() => {
        l += 2
        if (l > MODEL_CONFIG.num_hidden_layers) l = MODEL_CONFIG.num_hidden_layers
        setLayerCount(l)
        if (l >= MODEL_CONFIG.num_hidden_layers) clearInterval(iv)
      }, 30)
      return () => clearInterval(iv)
    }
  }, [currentStep])

  const reset = () => {
    setPlaying(false)
    setCurrentStep(0)
    setLayerCount(0)
    setTypedChars(0)
  }

  const currentStepData = PIPELINE_STEPS?.[currentStep]

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h2 className="text-2xl lg:text-3xl font-display font-bold tracking-tight">Inference Pipeline</h2>
          <p className="text-muted-foreground mt-1">Step-by-step token generation walkthrough</p>
        </div>
        <div className="flex items-center gap-2">
          <button onClick={reset} className="p-2 rounded-lg bg-muted/50 hover:bg-muted text-muted-foreground transition-colors">
            <RotateCcw className="w-4 h-4" />
          </button>
          <button
            onClick={() => setPlaying(!playing)}
            className="p-2 rounded-lg bg-primary/20 hover:bg-primary/30 text-primary transition-colors"
          >
            {playing ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
          </button>
          <button
            onClick={advanceStep}
            disabled={currentStep >= stepCount - 1}
            className="p-2 rounded-lg bg-muted/50 hover:bg-muted text-muted-foreground transition-colors disabled:opacity-30"
          >
            <SkipForward className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Timeline */}
      <div className="overflow-x-auto scrollbar-none">
        <div className="flex items-center gap-1 min-w-max py-2">
          {PIPELINE_STEPS?.map?.((step: any, i: number) => (
            <div key={step?.id} className="flex items-center">
              <button
                onClick={() => { setCurrentStep(i); setPlaying(false) }}
                className={`px-3 py-2 rounded-lg text-xs font-medium transition-all whitespace-nowrap ${
                  i === currentStep
                    ? 'bg-primary text-primary-foreground glow-blue'
                    : i < currentStep
                    ? 'bg-accent/20 text-accent'
                    : 'bg-muted/30 text-muted-foreground hover:bg-muted/50'
                }`}
              >
                {step?.title}
              </button>
              {i < stepCount - 1 && (
                <ChevronRight className={`w-3 h-3 mx-0.5 ${
                  i < currentStep ? 'text-accent' : 'text-muted-foreground/30'
                }`} />
              )}
            </div>
          )) ?? []}
        </div>
      </div>

      {/* Progress bar */}
      <div className="w-full h-1 rounded-full bg-muted/30 overflow-hidden">
        <motion.div
          className="h-full bg-gradient-to-r from-primary to-accent rounded-full"
          animate={{ width: `${((currentStep + 1) / stepCount) * 100}%` }}
          transition={{ duration: 0.3 }}
        />
      </div>

      {/* Step content */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Visualization panel */}
        <div className="glass-card rounded-xl p-6 border border-border/50 min-h-[280px] flex flex-col">
          <h3 className="text-sm font-mono text-muted-foreground uppercase tracking-wider mb-4">
            Step {currentStep + 1}: {currentStepData?.title}
          </h3>

          <div className="flex-1 flex items-center justify-center">
            {currentStep === 0 && (
              <div className="font-mono text-lg">
                <span className="text-primary">&gt; </span>
                <span>{EXAMPLE_PROMPT?.slice?.(0, typedChars) ?? ''}</span>
                <motion.span
                  className="inline-block w-2 h-5 bg-primary ml-0.5"
                  animate={{ opacity: [1, 0] }}
                  transition={{ duration: 0.6, repeat: Infinity }}
                />
              </div>
            )}

            {currentStep === 1 && (
              <div className="flex flex-wrap gap-2 justify-center">
                {EXAMPLE_TOKENS?.map?.((tok: any, i: number) => (
                  <motion.div
                    key={i}
                    initial={{ opacity: 0, scale: 0.5 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ delay: i * 0.15 }}
                    className="px-3 py-2 rounded-lg bg-primary/15 border border-primary/30 text-sm font-mono"
                  >
                    <span className="text-primary">{tok?.text}</span>
                    <span className="text-muted-foreground text-xs ml-2">#{tok?.id}</span>
                  </motion.div>
                )) ?? []}
              </div>
            )}

            {currentStep === 2 && (
              <div className="font-mono text-xs leading-relaxed text-center">
                <span className="text-accent">&lt;|im_start|&gt;</span><span className="text-muted-foreground">system</span><br />
                <span className="text-muted-foreground">You are a helpful assistant.</span><br />
                <span className="text-accent">&lt;|im_end|&gt;</span><br />
                <span className="text-accent">&lt;|im_start|&gt;</span><span className="text-muted-foreground">user</span><br />
                <span className="text-foreground">What is 2+2?</span><br />
                <span className="text-accent">&lt;|im_end|&gt;</span><br />
                <span className="text-accent">&lt;|im_start|&gt;</span><span className="text-primary">assistant</span>
              </div>
            )}

            {currentStep === 3 && (
              <div className="space-y-2 w-full">
                <p className="text-xs text-muted-foreground text-center mb-3">4096-dim embedding vectors</p>
                <div className="flex gap-1 justify-center">
                  {Array.from({ length: 32 }).map((_: unknown, i: number) => (
                    <motion.div
                      key={i}
                      initial={{ height: 4 }}
                      animate={{ height: 8 + (Math.sin(i * 0.7) * 0.5 + 0.5) * 40 }}
                      transition={{ delay: i * 0.02, duration: 0.4 }}
                      className="w-1.5 rounded-full bg-gradient-to-t from-primary/30 to-primary"
                    />
                  ))}
                </div>
                <p className="text-xs text-muted-foreground text-center">Token ID → vec[4096]</p>
              </div>
            )}

            {currentStep === 4 && (
              <div className="text-center space-y-3">
                <div className="text-4xl font-display font-bold text-primary glow-text-blue">
                  {layerCount} / {MODEL_CONFIG.num_hidden_layers}
                </div>
                <div className="w-full max-w-xs mx-auto h-2 rounded-full bg-muted/30 overflow-hidden">
                  <motion.div
                    className="h-full bg-gradient-to-r from-primary to-accent rounded-full"
                    animate={{ width: `${(layerCount / MODEL_CONFIG.num_hidden_layers) * 100}%` }}
                  />
                </div>
                <p className="text-xs text-muted-foreground">RMSNorm → Attention → RMSNorm → MoE FFN</p>
              </div>
            )}

            {currentStep === 5 && (
              <div className="space-y-3 w-full">
                <p className="text-xs text-muted-foreground text-center">GQA: 4 KV heads → 64 Q heads (16:1 broadcast)</p>
                <div className="flex items-center justify-center gap-4">
                  <div className="space-y-1">
                    <p className="text-xs font-mono text-accent">K,V (4)</p>
                    {[0,1,2,3].map((i: number) => (
                      <motion.div
                        key={i}
                        className="w-8 h-2 rounded bg-accent/60"
                        animate={{ opacity: [0.4, 1, 0.4] }}
                        transition={{ duration: 1.5, repeat: Infinity, delay: i * 0.2 }}
                      />
                    ))}
                  </div>
                  <div className="flex flex-col items-center gap-0.5">
                    {[0,1,2].map((i: number) => (
                      <motion.div
                        key={i}
                        className="w-8 h-px bg-primary/50"
                        animate={{ scaleX: [0, 1] }}
                        transition={{ duration: 0.5, delay: i * 0.2 }}
                      />
                    ))}
                    <span className="text-xs text-muted-foreground">×16</span>
                  </div>
                  <div className="space-y-0.5">
                    <p className="text-xs font-mono text-primary">Q (64)</p>
                    <div className="grid grid-cols-8 gap-px">
                      {Array.from({ length: 16 }).map((_: unknown, i: number) => (
                        <motion.div
                          key={i}
                          className="w-2 h-2 rounded-sm bg-primary/60"
                          animate={{ opacity: [0.3, 1, 0.3] }}
                          transition={{ duration: 1, repeat: Infinity, delay: i * 0.05 }}
                        />
                      ))}
                    </div>
                  </div>
                </div>
                <p className="text-xs text-muted-foreground text-center mt-2">softmax(Q·Kᵀ / √64) · V</p>
              </div>
            )}

            {currentStep === 6 && (
              <div className="w-full space-y-3">
                <p className="text-xs text-muted-foreground text-center">Gate → top-8 of 128 experts</p>
                <div className="grid grid-cols-16 gap-0.5 mx-auto" style={{ maxWidth: 280 }}>
                  {Array.from({ length: 128 }).map((_: unknown, i: number) => {
                    const isTop8 = [7, 23, 41, 55, 72, 89, 101, 118].includes(i)
                    return (
                      <motion.div
                        key={i}
                        className={`w-full aspect-square rounded-sm ${
                          isTop8 ? 'bg-primary glow-blue' : 'bg-muted/30'
                        }`}
                        initial={{ opacity: 0 }}
                        animate={{ opacity: isTop8 ? 1 : 0.4 }}
                        transition={{ delay: i * 0.005 }}
                      />
                    )
                  })}
                </div>
                <p className="text-xs text-muted-foreground text-center">Each expert: 768-wide FFN, 18 MB slab</p>
              </div>
            )}

            {currentStep === 7 && (
              <div className="w-full space-y-3">
                <p className="text-xs text-muted-foreground text-center">Temperature → top-k → top-p → sample</p>
                <div className="flex items-end gap-1 justify-center h-24">
                  {[0.02, 0.05, 0.35, 0.15, 0.08, 0.12, 0.03, 0.05, 0.04, 0.03, 0.02, 0.06].map((v: number, i: number) => (
                    <motion.div
                      key={i}
                      className={`w-5 rounded-t ${
                        i === 2 ? 'bg-primary glow-blue' : 'bg-muted-foreground/30'
                      }`}
                      initial={{ height: 0 }}
                      animate={{ height: v * 240 }}
                      transition={{ delay: i * 0.05, duration: 0.4 }}
                    />
                  ))}
                </div>
                <p className="text-xs text-center">
                  <span className="text-primary font-mono">token #1538</span>
                  <span className="text-muted-foreground"> sampled (p=0.35)</span>
                </p>
              </div>
            )}

            {currentStep === 8 && (
              <div className="text-center space-y-3">
                <motion.div
                  initial={{ scale: 0 }}
                  animate={{ scale: 1 }}
                  className="text-3xl font-display font-bold text-accent glow-text-cyan"
                >
                  "4"
                </motion.div>
                <p className="text-sm text-muted-foreground">Token appended to response</p>
                <p className="text-xs text-muted-foreground font-mono">
                  Stop token: {MODEL_CONFIG.stop_token} (〈|im_end|〉)
                </p>
              </div>
            )}
          </div>
        </div>

        {/* Description panel */}
        <div className="glass-card rounded-xl p-6 border border-border/50">
          <h3 className="font-display font-semibold text-lg mb-3">{currentStepData?.title}</h3>
          <p className="text-muted-foreground mb-4">{currentStepData?.description}</p>
          <div className="p-4 rounded-lg bg-muted/20 border border-border/30">
            <p className="text-xs font-mono text-muted-foreground uppercase tracking-wider mb-2">Technical Detail</p>
            <p className="text-sm text-muted-foreground leading-relaxed">{currentStepData?.detail}</p>
          </div>
        </div>
      </div>
    </div>
  )
}
