const DEFAULT_GRID = [15, 15]
const DEFAULT_NODE_SIZE = { width: 170, height: 90 }
const VIEWPORT_BOUNDS = { minX: -2000, minY: -2000, maxX: 5000, maxY: 5000 }

function toFiniteNumber(value, fallback) {
  const n = Number(value)
  return Number.isFinite(n) ? n : fallback
}

export function getNodeSize(node, sizeHint) {
  return {
    width: toFiniteNumber(sizeHint?.width ?? node?.width ?? node?.measured?.width, DEFAULT_NODE_SIZE.width),
    height: toFiniteNumber(sizeHint?.height ?? node?.height ?? node?.measured?.height, DEFAULT_NODE_SIZE.height),
  }
}

export function clampToViewport(position, nodeSize, bounds = VIEWPORT_BOUNDS) {
  const w = nodeSize?.width || DEFAULT_NODE_SIZE.width
  const h = nodeSize?.height || DEFAULT_NODE_SIZE.height
  return {
    x: Math.max(bounds.minX, Math.min(bounds.maxX - w, position.x)),
    y: Math.max(bounds.minY, Math.min(bounds.maxY - h, position.y)),
  }
}

export function snapPositionToGrid(position, grid = DEFAULT_GRID) {
  const [gx, gy] = grid
  const safeX = gx > 0 ? gx : DEFAULT_GRID[0]
  const safeY = gy > 0 ? gy : DEFAULT_GRID[1]
  return {
    x: Math.round((position?.x || 0) / safeX) * safeX,
    y: Math.round((position?.y || 0) / safeY) * safeY,
  }
}

function overlapsRect(a, b, paddingX, paddingY) {
  const ax1 = a.position.x - paddingX
  const ay1 = a.position.y - paddingY
  const ax2 = a.position.x + a.size.width + paddingX
  const ay2 = a.position.y + a.size.height + paddingY

  const bx1 = b.position.x - paddingX
  const by1 = b.position.y - paddingY
  const bx2 = b.position.x + b.size.width + paddingX
  const by2 = b.position.y + b.size.height + paddingY

  return ax1 < bx2 && ax2 > bx1 && ay1 < by2 && ay2 > by1
}

function collides(candidate, occupied, paddingX, paddingY) {
  for (const placed of occupied) {
    if (overlapsRect(candidate, placed, paddingX, paddingY)) return true
  }
  return false
}

function findFreePosition(candidate, occupied, grid, paddingX, paddingY, maxRadius) {
  const snapped = snapPositionToGrid(candidate.position, grid)
  const base = { ...candidate, position: snapped }

  if (!collides(base, occupied, paddingX, paddingY)) {
    return snapped
  }

  const [gx, gy] = grid
  for (let radius = 1; radius <= maxRadius; radius += 1) {
    for (let ix = -radius; ix <= radius; ix += 1) {
      for (let iy = -radius; iy <= radius; iy += 1) {
        if (Math.abs(ix) !== radius && Math.abs(iy) !== radius) continue
        const next = {
          x: snapped.x + ix * gx,
          y: snapped.y + iy * gy,
        }
        const probe = { ...candidate, position: next }
        if (!collides(probe, occupied, paddingX, paddingY)) {
          return next
        }
      }
    }
  }

  return snapped
}

export function findNearestFreePosition(nodeId, desiredPosition, nodes, options = {}) {
  const grid = options.grid || DEFAULT_GRID
  const paddingX = toFiniteNumber(options.paddingX, 24)
  const paddingY = toFiniteNumber(options.paddingY, 18)
  const maxRadius = toFiniteNumber(options.maxRadius, 28)
  const target = nodes.find((n) => n.id === nodeId)
  if (!target) {
    const fallbackSize = options.sizeHint || DEFAULT_NODE_SIZE
    return clampToViewport(snapPositionToGrid(desiredPosition, grid), fallbackSize)
  }

  const occupied = nodes
    .filter((n) => n.id !== nodeId)
    .map((n) => ({ position: n.position, size: getNodeSize(n) }))

  const sizeHint = options.sizeHint || undefined
  const size = getNodeSize(target, sizeHint)
  const result = findFreePosition(
    { position: desiredPosition, size },
    occupied,
    grid,
    paddingX,
    paddingY,
    maxRadius
  )
  return clampToViewport(result, size)
}

export function normalizeNodePlacement(nodes, options = {}) {
  const grid = options.grid || DEFAULT_GRID
  const paddingX = toFiniteNumber(options.paddingX, 24)
  const paddingY = toFiniteNumber(options.paddingY, 18)
  const maxRadius = toFiniteNumber(options.maxRadius, 20)
  const placed = []

  return nodes.map((node) => {
    const candidate = { position: node.position, size: getNodeSize(node) }
    const free = findFreePosition(candidate, placed, grid, paddingX, paddingY, maxRadius)
    const resolved = { ...node, position: free }
    placed.push({ position: free, size: getNodeSize(resolved) })
    return resolved
  })
}

function asIdSet(items) {
  const out = new Set()
  for (const item of items || []) out.add(typeof item === 'string' ? item : item.id)
  return out
}

export function alignNodesHorizontally(nodes, targetNodeIds, options = {}) {
  const selected = asIdSet(targetNodeIds)
  if (selected.size === 0) return nodes
  const gap = toFiniteNumber(options.gap, 26)
  const candidates = nodes.filter((n) => selected.has(n.id))
  if (candidates.length < 2) return nodes

  const baselineY = Math.round(
    candidates.reduce((acc, n) => acc + n.position.y + getNodeSize(n).height / 2, 0) / candidates.length
  )
  const sorted = [...candidates].sort((a, b) => a.position.x - b.position.x)
  let cursorX = sorted[0].position.x

  const nextPos = new Map()
  for (let i = 0; i < sorted.length; i += 1) {
    const n = sorted[i]
    const sz = getNodeSize(n)
    const yTop = baselineY - sz.height / 2
    const snapped = snapPositionToGrid({ x: cursorX, y: yTop })
    nextPos.set(n.id, snapped)
    cursorX = snapped.x + sz.width + gap
  }

  const remapped = nodes.map((n) => (nextPos.has(n.id) ? { ...n, position: nextPos.get(n.id) } : n))
  return normalizeNodePlacement(remapped, options)
}

export function alignNodesVertically(nodes, targetNodeIds, options = {}) {
  const selected = asIdSet(targetNodeIds)
  if (selected.size === 0) return nodes
  const gap = toFiniteNumber(options.gap, 22)
  const candidates = nodes.filter((n) => selected.has(n.id))
  if (candidates.length < 2) return nodes

  const baselineX = Math.round(
    candidates.reduce((acc, n) => acc + n.position.x + getNodeSize(n).width / 2, 0) / candidates.length
  )
  const sorted = [...candidates].sort((a, b) => a.position.y - b.position.y)
  let cursorY = sorted[0].position.y

  const nextPos = new Map()
  for (let i = 0; i < sorted.length; i += 1) {
    const n = sorted[i]
    const sz = getNodeSize(n)
    const xLeft = baselineX - sz.width / 2
    const snapped = snapPositionToGrid({ x: xLeft, y: cursorY })
    nextPos.set(n.id, snapped)
    cursorY = snapped.y + sz.height + gap
  }

  const remapped = nodes.map((n) => (nextPos.has(n.id) ? { ...n, position: nextPos.get(n.id) } : n))
  return normalizeNodePlacement(remapped, options)
}

export function distributeNodesHorizontally(nodes, targetNodeIds, options = {}) {
  const selected = asIdSet(targetNodeIds)
  if (selected.size === 0) return nodes
  const minGap = toFiniteNumber(options.minGap, 24)
  const candidates = nodes.filter((n) => selected.has(n.id))
  if (candidates.length < 3) return alignNodesHorizontally(nodes, targetNodeIds, options)

  const sorted = [...candidates].sort((a, b) => a.position.x - b.position.x)
  const first = sorted[0]
  const last = sorted[sorted.length - 1]
  const totalWidth = sorted.reduce((acc, n) => acc + getNodeSize(n).width, 0)
  const extent = (last.position.x + getNodeSize(last).width) - first.position.x
  const naturalGap = (extent - totalWidth) / (sorted.length - 1)
  const gap = Number.isFinite(naturalGap) ? Math.max(minGap, naturalGap) : minGap

  let cursorX = first.position.x
  const nextPos = new Map()
  for (let i = 0; i < sorted.length; i += 1) {
    const n = sorted[i]
    const snapped = snapPositionToGrid({ x: cursorX, y: n.position.y })
    nextPos.set(n.id, snapped)
    cursorX = snapped.x + getNodeSize(n).width + gap
  }

  const remapped = nodes.map((n) => (nextPos.has(n.id) ? { ...n, position: nextPos.get(n.id) } : n))
  return normalizeNodePlacement(remapped, options)
}

export function distributeNodesVertically(nodes, targetNodeIds, options = {}) {
  const selected = asIdSet(targetNodeIds)
  if (selected.size === 0) return nodes
  const minGap = toFiniteNumber(options.minGap, 20)
  const candidates = nodes.filter((n) => selected.has(n.id))
  if (candidates.length < 3) return alignNodesVertically(nodes, targetNodeIds, options)

  const sorted = [...candidates].sort((a, b) => a.position.y - b.position.y)
  const first = sorted[0]
  const last = sorted[sorted.length - 1]
  const totalHeight = sorted.reduce((acc, n) => acc + getNodeSize(n).height, 0)
  const extent = (last.position.y + getNodeSize(last).height) - first.position.y
  const naturalGap = (extent - totalHeight) / (sorted.length - 1)
  const gap = Number.isFinite(naturalGap) ? Math.max(minGap, naturalGap) : minGap

  let cursorY = first.position.y
  const nextPos = new Map()
  for (let i = 0; i < sorted.length; i += 1) {
    const n = sorted[i]
    const snapped = snapPositionToGrid({ x: n.position.x, y: cursorY })
    nextPos.set(n.id, snapped)
    cursorY = snapped.y + getNodeSize(n).height + gap
  }

  const remapped = nodes.map((n) => (nextPos.has(n.id) ? { ...n, position: nextPos.get(n.id) } : n))
  return normalizeNodePlacement(remapped, options)
}

export function tidySelectedNodes(nodes, targetNodeIds, options = {}) {
  const selected = asIdSet(targetNodeIds)
  if (selected.size === 0) return nodes
  const grid = options.grid || DEFAULT_GRID
  const paddingX = toFiniteNumber(options.paddingX, 24)
  const paddingY = toFiniteNumber(options.paddingY, 18)
  const maxRadius = toFiniteNumber(options.maxRadius, 24)

  const fixed = nodes
    .filter((n) => !selected.has(n.id))
    .map((n) => ({ position: n.position, size: getNodeSize(n) }))

  const moving = nodes
    .filter((n) => selected.has(n.id))
    .sort((a, b) => {
      if (a.position.y !== b.position.y) return a.position.y - b.position.y
      return a.position.x - b.position.x
    })

  const placed = [...fixed]
  const out = new Map()
  for (const node of moving) {
    const candidate = {
      position: snapPositionToGrid(node.position, grid),
      size: getNodeSize(node),
    }
    const free = findFreePosition(candidate, placed, grid, paddingX, paddingY, maxRadius)
    out.set(node.id, free)
    placed.push({ position: free, size: candidate.size })
  }

  return nodes.map((n) => (out.has(n.id) ? { ...n, position: out.get(n.id) } : n))
}
