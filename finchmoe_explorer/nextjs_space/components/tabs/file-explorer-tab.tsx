'use client'

import { useState, useMemo } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Search, FileCode2, ChevronDown, ChevronRight, X, Hash, Code2 } from 'lucide-react'
import { SOURCE_FILES, SESSION_LABELS, TOTAL_FILES, TOTAL_LINES, TOTAL_SESSIONS } from '@/lib/data'
import type { SourceFile } from '@/lib/data'

export default function FileExplorerTab() {
  const [search, setSearch] = useState('')
  const [expandedSessions, setExpandedSessions] = useState<Set<number>>(new Set([1, 2, 3, 4, 5, 0]))
  const [selectedFile, setSelectedFile] = useState<SourceFile | null>(null)

  const filteredFiles = useMemo(() => {
    if (!search?.trim?.()) return SOURCE_FILES ?? []
    const q = search?.toLowerCase?.() ?? ''
    return (SOURCE_FILES ?? []).filter((f: SourceFile) =>
      (f?.name?.toLowerCase?.() ?? '').includes(q) ||
      (f?.description?.toLowerCase?.() ?? '').includes(q) ||
      (f?.symbols ?? []).some((s: string) => (s?.toLowerCase?.() ?? '').includes(q))
    )
  }, [search])

  const groupedFiles = useMemo(() => {
    const groups: Record<number, SourceFile[]> = {}
    ;(filteredFiles ?? []).forEach((f: SourceFile) => {
      const sess = f?.session ?? 0
      if (!groups[sess]) groups[sess] = []
      groups[sess].push(f)
    })
    return groups
  }, [filteredFiles])

  const toggleSession = (session: number) => {
    setExpandedSessions((prev: Set<number>) => {
      const next = new Set(prev)
      if (next.has(session)) next.delete(session)
      else next.add(session)
      return next
    })
  }

  const typeIcon = (type: string) => {
    switch (type) {
      case 'header': return 'H'
      case 'source': return 'S'
      case 'metal': return 'M'
      case 'cmake': return 'C'
      case 'markdown': return 'D'
      case 'test': return 'T'
      default: return '?'
    }
  }

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl lg:text-3xl font-display font-bold tracking-tight">File Explorer</h2>
        <p className="text-muted-foreground mt-1">Complete source map of the FinchMoE engine</p>
      </div>

      {/* Stats */}
      <div className="flex items-center gap-3 text-sm">
        <span className="px-3 py-1.5 rounded-lg bg-muted/30 border border-border/50 font-mono text-muted-foreground">
          {TOTAL_FILES} files
        </span>
        <span className="px-3 py-1.5 rounded-lg bg-muted/30 border border-border/50 font-mono text-muted-foreground">
          {TOTAL_LINES?.toLocaleString?.('en-US') ?? '0'} lines
        </span>
        <span className="px-3 py-1.5 rounded-lg bg-muted/30 border border-border/50 font-mono text-muted-foreground">
          {TOTAL_SESSIONS} sessions
        </span>
      </div>

      {/* Search */}
      <div className="relative">
        <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
        <input
          type="text"
          placeholder="Search files, symbols, descriptions..."
          value={search}
          onChange={(e: React.ChangeEvent<HTMLInputElement>) => setSearch(e?.target?.value ?? '')}
          className="w-full pl-10 pr-10 py-2.5 rounded-lg bg-muted/20 border border-border/50 text-sm placeholder:text-muted-foreground/50 focus:outline-none focus:ring-1 focus:ring-primary/50 focus:border-primary/30"
        />
        {search && (
          <button
            onClick={() => setSearch('')}
            className="absolute right-3 top-1/2 -translate-y-1/2 p-0.5 rounded hover:bg-muted/50"
          >
            <X className="w-3 h-3 text-muted-foreground" />
          </button>
        )}
      </div>

      <div className="flex flex-col lg:flex-row gap-6">
        {/* File tree */}
        <div className="flex-1 space-y-2">
          {Object.entries(groupedFiles ?? {}).sort(([a]: any, [b]: any) => Number(a) - Number(b)).map(([sessionKey, files]: any) => {
            const session = Number(sessionKey)
            const info = SESSION_LABELS?.[session]
            const isExpanded = expandedSessions?.has?.(session)
            return (
              <div key={session} className="glass-card rounded-xl border border-border/50 overflow-hidden">
                <button
                  onClick={() => toggleSession(session)}
                  className="w-full flex items-center gap-3 px-4 py-3 hover:bg-muted/20 transition-colors"
                >
                  {isExpanded ? <ChevronDown className="w-4 h-4 text-muted-foreground" /> : <ChevronRight className="w-4 h-4 text-muted-foreground" />}
                  <span
                    className="w-2 h-2 rounded-full"
                    style={{ backgroundColor: info?.color ?? '#64748b' }}
                  />
                  <span className="text-sm font-medium">
                    {session === 0 ? 'Build & Tests' : `Session ${session}: ${info?.title ?? ''}`}
                  </span>
                  <span className="text-xs text-muted-foreground ml-auto font-mono">
                    {(files as SourceFile[])?.length ?? 0} files
                  </span>
                </button>

                <AnimatePresence>
                  {isExpanded && (
                    <motion.div
                      initial={{ height: 0, opacity: 0 }}
                      animate={{ height: 'auto', opacity: 1 }}
                      exit={{ height: 0, opacity: 0 }}
                      transition={{ duration: 0.2 }}
                      className="overflow-hidden"
                    >
                      <div className="px-2 pb-2 space-y-0.5">
                        {(files as SourceFile[])?.map?.((file: SourceFile) => (
                          <button
                            key={file?.name}
                            onClick={() => setSelectedFile(selectedFile?.name === file?.name ? null : file)}
                            className={`w-full flex items-center gap-2 px-3 py-2 rounded-lg text-sm transition-colors ${
                              selectedFile?.name === file?.name
                                ? 'bg-primary/15 text-primary'
                                : 'text-muted-foreground hover:text-foreground hover:bg-muted/20'
                            }`}
                          >
                            <span className="w-5 h-5 rounded text-[9px] font-mono font-bold flex items-center justify-center bg-muted/40 text-muted-foreground flex-shrink-0">
                              {typeIcon(file?.type ?? '')}
                            </span>
                            <FileCode2 className="w-3.5 h-3.5 flex-shrink-0" />
                            <span className="font-mono text-xs truncate">{file?.name}</span>
                            <span className="text-xs text-muted-foreground/60 ml-auto font-mono">{file?.lines}L</span>
                          </button>
                        )) ?? []}
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
            )
          })}
        </div>

        {/* File detail drawer */}
        <AnimatePresence>
          {selectedFile && (
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 20 }}
              className="w-full lg:w-80 xl:w-96"
            >
              <div className="glass-card rounded-xl p-5 border border-border/50 sticky top-24">
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center gap-2">
                    <FileCode2 className="w-5 h-5 text-primary" />
                    <h3 className="font-mono font-bold text-sm">{selectedFile?.name}</h3>
                  </div>
                  <button onClick={() => setSelectedFile(null)} className="p-1 rounded hover:bg-muted/50">
                    <X className="w-4 h-4 text-muted-foreground" />
                  </button>
                </div>

                <div className="space-y-4">
                  <div className="flex flex-wrap gap-2">
                    <span className="px-2 py-1 rounded text-xs font-mono bg-muted/40 text-muted-foreground">
                      {selectedFile?.type}
                    </span>
                    <span className="px-2 py-1 rounded text-xs font-mono bg-muted/40 text-muted-foreground">
                      <Hash className="w-3 h-3 inline mr-1" />
                      {selectedFile?.lines} lines
                    </span>
                    <span
                      className="px-2 py-1 rounded text-xs font-mono"
                      style={{
                        backgroundColor: `${SESSION_LABELS?.[selectedFile?.session ?? 0]?.color ?? '#64748b'}20`,
                        color: SESSION_LABELS?.[selectedFile?.session ?? 0]?.color ?? '#64748b',
                      }}
                    >
                      Session {selectedFile?.session}
                    </span>
                  </div>

                  <div>
                    <p className="text-xs font-mono text-muted-foreground uppercase tracking-wider mb-1">Path</p>
                    <p className="text-sm font-mono text-foreground">{selectedFile?.path ?? ''}{selectedFile?.name}</p>
                  </div>

                  <div>
                    <p className="text-xs font-mono text-muted-foreground uppercase tracking-wider mb-1">Description</p>
                    <p className="text-sm text-muted-foreground leading-relaxed">{selectedFile?.description}</p>
                  </div>

                  <div>
                    <p className="text-xs font-mono text-muted-foreground uppercase tracking-wider mb-2">Key Symbols</p>
                    <div className="flex flex-wrap gap-1.5">
                      {selectedFile?.symbols?.map?.((sym: string, i: number) => (
                        <span
                          key={i}
                          className="flex items-center gap-1 px-2 py-1 rounded text-xs font-mono bg-primary/10 text-primary border border-primary/20"
                        >
                          <Code2 className="w-3 h-3" />
                          {sym}
                        </span>
                      )) ?? []}
                    </div>
                  </div>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  )
}
