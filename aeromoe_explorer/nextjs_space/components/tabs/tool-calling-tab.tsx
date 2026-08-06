'use client'

import { useState, useEffect, useRef } from 'react'
import { motion } from 'framer-motion'
import { Play, RotateCcw, Code2, ArrowDown, CheckCircle2, Loader2 } from 'lucide-react'
import { TOOL_CALL_DEMO_STEPS } from '@/lib/data'

const FLOW_NODES = [
  { id: 'user', label: 'User Prompt', y: 0 },
  { id: 'format', label: 'Format ChatML', y: 1 },
  { id: 'generate', label: 'InferenceEngine::generate()', y: 2 },
  { id: 'check', label: 'Contains <tool_call>?', y: 3 },
  { id: 'parse', label: 'ToolParser::parse()', y: 4 },
  { id: 'dispatch', label: 'Dispatch to handler', y: 5 },
  { id: 'append', label: 'Append ToolResult', y: 6 },
  { id: 'output', label: 'Return final answer', y: 3.5 },
]

const CODE_SNIPPET = `// C++ Tool Registration API
auto& registry = engine.tool_registry();

registry.register("calculator", {
  .description = "Evaluate a math expression",
  .parameters = {{"expression", "string", true}},
  .handler = [](const json& args) -> ToolResult {
    return evaluate(args["expression"]);
  }
});

registry.register("get_time", {
  .description = "Get current date and time",
  .handler = [](const json&) -> ToolResult {
    return current_time();
  }
});`

export default function ToolCallingTab() {
  const [currentDemoStep, setCurrentDemoStep] = useState(-1)
  const [playing, setPlaying] = useState(false)
  const timerRef = useRef<NodeJS.Timeout | null>(null)

  const totalSteps = TOOL_CALL_DEMO_STEPS?.length ?? 0

  useEffect(() => {
    if (playing && currentDemoStep < totalSteps - 1) {
      timerRef.current = setTimeout(() => {
        setCurrentDemoStep((p: number) => p + 1)
      }, 1500)
    } else if (currentDemoStep >= totalSteps - 1) {
      setPlaying(false)
    }
    return () => {
      if (timerRef?.current) clearTimeout(timerRef.current)
    }
  }, [playing, currentDemoStep, totalSteps])

  const startDemo = () => {
    setCurrentDemoStep(0)
    setPlaying(true)
  }

  const resetDemo = () => {
    setPlaying(false)
    setCurrentDemoStep(-1)
  }

  const currentNode = TOOL_CALL_DEMO_STEPS?.[currentDemoStep]?.node ?? ''

  const typeColor = (type: string) => {
    switch (type) {
      case 'user': return 'text-primary'
      case 'model': return 'text-accent'
      case 'system': return 'text-muted-foreground'
      case 'tool': return 'text-green-400'
      case 'final': return 'text-yellow-400'
      default: return 'text-foreground'
    }
  }

  const typeBg = (type: string) => {
    switch (type) {
      case 'user': return 'bg-primary/10 border-primary/30'
      case 'model': return 'bg-accent/10 border-accent/30'
      case 'system': return 'bg-muted/30 border-border/30'
      case 'tool': return 'bg-green-500/10 border-green-500/30'
      case 'final': return 'bg-yellow-500/10 border-yellow-500/30'
      default: return 'bg-muted/10 border-border/30'
    }
  }

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h2 className="text-2xl lg:text-3xl font-display font-bold tracking-tight">Tool-Calling Loop</h2>
          <p className="text-muted-foreground mt-1">Agentic multi-turn tool execution flow</p>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={startDemo}
            disabled={playing}
            className="flex items-center gap-2 px-4 py-2 rounded-lg bg-primary text-primary-foreground font-medium text-sm hover:bg-primary/90 transition-colors disabled:opacity-50 glow-blue"
          >
            <Play className="w-4 h-4" /> Play Demo
          </button>
          <button
            onClick={resetDemo}
            className="p-2 rounded-lg bg-muted/50 hover:bg-muted text-muted-foreground transition-colors"
          >
            <RotateCcw className="w-4 h-4" />
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Flowchart */}
        <div className="glass-card rounded-xl p-5 border border-border/50">
          <p className="text-xs font-mono text-muted-foreground uppercase tracking-wider mb-4">Flow Diagram</p>
          <div className="space-y-2">
            {FLOW_NODES?.map?.((node: any, i: number) => {
              const isActive = currentNode === node?.id
              const isPast = TOOL_CALL_DEMO_STEPS?.slice?.(0, currentDemoStep + 1)
                ?.some?.((s: any) => s?.node === node?.id)
              return (
                <div key={node?.id}>
                  <div
                    className={`px-3 py-2 rounded-lg text-xs font-mono text-center transition-all duration-300 ${
                      isActive
                        ? 'bg-primary/20 border border-primary/50 text-primary glow-blue'
                        : isPast
                        ? 'bg-accent/10 border border-accent/30 text-accent'
                        : 'bg-muted/20 border border-border/30 text-muted-foreground'
                    }`}
                  >
                    {node?.id === 'check' ? (
                      <span>
                        Contains &lt;tool_call&gt;?
                        {isActive && <span className="ml-2 text-accent">YES</span>}
                      </span>
                    ) : (
                      node?.label
                    )}
                  </div>
                  {i < (FLOW_NODES?.length ?? 0) - 1 && (
                    <div className="flex justify-center py-0.5">
                      <ArrowDown className={`w-3 h-3 ${
                        isPast ? 'text-accent' : 'text-muted-foreground/30'
                      }`} />
                    </div>
                  )}
                </div>
              )
            }) ?? []}
          </div>
        </div>

        {/* Demo output */}
        <div className="glass-card rounded-xl p-5 border border-border/50">
          <p className="text-xs font-mono text-muted-foreground uppercase tracking-wider mb-4">Demo Execution</p>
          <div className="space-y-2 max-h-[500px] overflow-y-auto scrollbar-none">
            {currentDemoStep < 0 ? (
              <p className="text-sm text-muted-foreground">Click "Play Demo" to start</p>
            ) : (
              TOOL_CALL_DEMO_STEPS?.slice?.(0, currentDemoStep + 1)?.map?.((step: any, i: number) => (
                <motion.div
                  key={step?.id}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className={`p-3 rounded-lg border ${typeBg(step?.type)}`}
                >
                  <div className="flex items-center gap-2 mb-1">
                    {i === currentDemoStep && playing ? (
                      <Loader2 className="w-3 h-3 text-primary animate-spin" />
                    ) : (
                      <CheckCircle2 className={`w-3 h-3 ${typeColor(step?.type)}`} />
                    )}
                    <span className={`text-xs font-medium ${typeColor(step?.type)}`}>
                      {step?.label}
                    </span>
                  </div>
                  <pre className="text-xs font-mono text-muted-foreground whitespace-pre-wrap break-all leading-relaxed">
                    {step?.content}
                  </pre>
                </motion.div>
              )) ?? []
            )}
          </div>
        </div>

        {/* Code snippet */}
        <div className="glass-card rounded-xl p-5 border border-border/50">
          <div className="flex items-center gap-2 mb-4">
            <Code2 className="w-4 h-4 text-primary" />
            <p className="text-xs font-mono text-muted-foreground uppercase tracking-wider">Registration API</p>
          </div>
          <pre className="text-xs font-mono leading-relaxed text-muted-foreground overflow-x-auto scrollbar-none whitespace-pre">
            <code>{CODE_SNIPPET}</code>
          </pre>
        </div>
      </div>
    </div>
  )
}
