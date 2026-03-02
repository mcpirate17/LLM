/**
 * Shared category icons, colors, and labels.
 * Single source of truth — import into DesignerNode, Inspector, RunResultsPanel, etc.
 */
import {
  ArrowUpDown, Sigma, Grid3X3, Puzzle, Waves, Activity,
  FunctionSquare, Shuffle, Layers, GitFork, Scale, Ruler,
  Hexagon, Box, Compass, Database, Filter, Repeat, Orbit,
} from 'lucide-react'

export const CATEGORY_ICONS = {
  io: ArrowUpDown,
  data_io: Database,
  data_transform: Filter,
  control_flow: Repeat,
  math: Sigma,
  linear_algebra: Grid3X3,
  mixing: Shuffle,
  channel_mixing: Layers,
  sequence: Waves,
  frequency: Activity,
  normalization: Scale,
  positional: Ruler,
  structural: Puzzle,
  representation: Hexagon,
  routing: GitFork,
  topology: Compass,
  blocks: Box,
  functional: FunctionSquare,
  math_space: Orbit,
}

export const CATEGORY_COLORS = {
  io: '#17a3ff',
  math: '#24d1a0',
  linear_algebra: '#a060ff',
  structural: '#f0a020',
  sequence: '#ff6090',
  frequency: '#20c0f0',
  functional: '#c060c0',
  mixing: '#ff8040',
  channel_mixing: '#e0c040',
  routing: '#60c060',
  normalization: '#8090ff',
  positional: '#ff60a0',
  representation: '#40d0d0',
  topology: '#d09040',
  blocks: '#90c0ff',
  math_space: '#2bd9a9',
  data_io: '#1aa5ff',
  data_transform: '#ff9a3d',
  control_flow: '#8bdc65',
}
