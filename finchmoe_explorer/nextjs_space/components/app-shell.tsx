'use client'

import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Cpu, GitBranch, Grid3X3, Gauge, Wrench, FolderTree, Menu, X } from 'lucide-react'
import { MODEL_CONFIG, TOTAL_FILES, TOTAL_LINES, TOTAL_SESSIONS } from '@/lib/data'

import OverviewTab from '@/components/tabs/overview-tab'
import PipelineTab from '@/components/tabs/pipeline-tab'
import MoeRoutingTab from '@/components/tabs/moe-routing-tab'
import MemoryTab from '@/components/tabs/memory-tab'
import ToolCallingTab from '@/components/tabs/tool-calling-tab'
import FileExplorerTab from '@/components/tabs/file-explorer-tab'

const TABS = [
  { id: 'overview', label: 'Overview', icon: Cpu },
  { id: 'pipeline', label: 'Inference Pipeline', icon: GitBranch },
  { id: 'moe', label: 'MoE Routing', icon: Grid3X3 },
  { id: 'memory', label: 'Memory Manager', icon: Gauge },
  { id: 'tools', label: 'Tool-Calling', icon: Wrench },
  { id: 'files', label: 'File Explorer', icon: FolderTree },
] as const

type TabId = typeof TABS[number]['id']

const STATS = [
  `${MODEL_CONFIG.num_hidden_layers} layers`,
  `${MODEL_CONFIG.num_experts} experts/layer`,
  `top-${MODEL_CONFIG.num_experts_per_tok} routing`,
  `${MODEL_CONFIG.cache_budget_gb} GB RAM cap`,
  'bf16',
]

export default function AppShell() {
  const [activeTab, setActiveTab] = useState<TabId>('overview')
  const [sidebarOpen, setSidebarOpen] = useState(false)

  const renderTab = () => {
    switch (activeTab) {
      case 'overview': return <OverviewTab />
      case 'pipeline': return <PipelineTab />
      case 'moe': return <MoeRoutingTab />
      case 'memory': return <MemoryTab />
      case 'tools': return <ToolCallingTab />
      case 'files': return <FileExplorerTab />
      default: return <OverviewTab />
    }
  }

  return (
    <div className="min-h-screen flex flex-col">
      {/* Header */}
      <header className="sticky top-0 z-50 glass-card border-b border-border/50">
        <div className="px-4 lg:px-6 py-3 flex items-center gap-4">
          {/* Mobile menu */}
          <button
            className="lg:hidden p-2 rounded-lg bg-muted/50 hover:bg-muted transition-colors"
            onClick={() => setSidebarOpen(!sidebarOpen)}
          >
            {sidebarOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
          </button>

          {/* Logo */}
          <div className="flex items-center gap-3">
            <div className="relative">
              <Cpu className="w-8 h-8 text-primary glow-text-blue" />
              <div className="absolute inset-0 blur-lg bg-primary/30 rounded-full" />
            </div>
            <div>
              <h1 className="text-xl font-display font-bold tracking-tight text-foreground glow-text-blue">
                FinchMoE
              </h1>
              <p className="text-xs text-muted-foreground font-mono hidden sm:block">
                Qwen3.6-35B-A3B · Apple Silicon · Metal GPU
              </p>
            </div>
          </div>

          {/* Stats chips */}
          <div className="hidden md:flex items-center gap-2 ml-auto">
            {STATS?.map?.((stat: string, i: number) => (
              <span
                key={i}
                className="px-2 py-1 text-xs font-mono rounded-full bg-muted/50 text-muted-foreground border border-border/50"
              >
                {stat}
              </span>
            )) ?? []}
          </div>
        </div>
      </header>

      <div className="flex flex-1 overflow-hidden">
        {/* Sidebar */}
        <AnimatePresence>
          {(sidebarOpen || true) && (
            <motion.aside
              initial={{ x: -280 }}
              animate={{ x: 0 }}
              className={`${
                sidebarOpen ? 'fixed inset-0 z-40 lg:relative lg:z-auto' : 'hidden lg:flex'
              } flex-col w-64 lg:w-56 xl:w-64 bg-card/80 backdrop-blur-xl border-r border-border/50 overflow-y-auto`}
            >
              {/* Mobile overlay */}
              {sidebarOpen && (
                <div
                  className="fixed inset-0 bg-black/50 lg:hidden -z-10"
                  onClick={() => setSidebarOpen(false)}
                />
              )}
              <nav className="flex flex-col gap-1 p-3 pt-4">
                <div className="px-3 py-2 mb-2">
                  <p className="text-xs font-mono text-muted-foreground uppercase tracking-wider">Navigation</p>
                </div>
                {TABS?.map?.((tab) => {
                  const Icon = tab?.icon
                  const isActive = activeTab === tab?.id
                  return (
                    <button
                      key={tab?.id}
                      onClick={() => {
                        setActiveTab(tab?.id as TabId)
                        setSidebarOpen(false)
                      }}
                      className={`flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-all duration-200 ${
                        isActive
                          ? 'bg-primary/15 text-primary glow-blue border border-primary/30'
                          : 'text-muted-foreground hover:text-foreground hover:bg-muted/50'
                      }`}
                    >
                      {Icon && <Icon className="w-4 h-4 flex-shrink-0" />}
                      <span>{tab?.label}</span>
                    </button>
                  )
                }) ?? []}

                <div className="mt-6 mx-3 p-3 rounded-lg bg-muted/30 border border-border/50">
                  <p className="text-xs font-mono text-muted-foreground">
                    {TOTAL_FILES} files · {TOTAL_LINES?.toLocaleString?.('en-US') ?? '0'} lines · {TOTAL_SESSIONS} sessions
                  </p>
                </div>
              </nav>
            </motion.aside>
          )}
        </AnimatePresence>

        {/* Main content */}
        <main className="flex-1 overflow-y-auto overflow-x-hidden p-4 lg:p-6">
          <AnimatePresence mode="wait">
            <motion.div
              key={activeTab}
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -12 }}
              transition={{ duration: 0.25 }}
              className="max-w-7xl mx-auto"
            >
              {renderTab()}
            </motion.div>
          </AnimatePresence>
        </main>
      </div>
    </div>
  )
}
