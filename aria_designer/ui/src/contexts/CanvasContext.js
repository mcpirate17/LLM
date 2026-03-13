import { createContext, useContext } from 'react'

/** Provides canvas-level props to DesignerNode without destabilising nodeTypes. */
export const CanvasContext = createContext({
  hardwareView: false,
  heatmapView: false,
  maxFlops: 0,
  openNodeHelp: () => {},
})

export const useCanvasContext = () => useContext(CanvasContext)
