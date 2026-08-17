/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Absolute API origin. Empty in dev, where vite proxies /api. */
  readonly VITE_API_BASE?: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
