/**
 * Visual check for the portal. The palette validator checks colour; nothing
 * checks layout, so this renders every view in both themes and fails on any
 * console error.
 *
 *   node portal/screenshot.mjs [outputPrefix]
 *
 * Requires the dev server (`npm run dev`) and the API to be running.
 */
import { chromium } from 'playwright'

const out = process.argv[2] ?? '/tmp/edgesense'
const url = process.env.PORTAL_URL ?? 'http://localhost:5173/'

const browser = await chromium.launch()
let failures = 0

for (const theme of ['light', 'dark']) {
  const context = await browser.newContext({
    viewport: { width: 1440, height: 1000 },
    colorScheme: theme,
  })
  const page = await context.newPage()
  const errors = []
  page.on('console', (m) => m.type() === 'error' && errors.push(m.text()))
  page.on('pageerror', (e) => errors.push(`PAGEERROR: ${e.message}`))

  await page.goto(url, { waitUntil: 'networkidle' })

  for (const [tab, name] of [
    ['Live call', 'live'],
    ['Supervisor', 'dashboard'],
    ['Latency', 'latency'],
  ]) {
    await page.getByRole('button', { name: tab }).click()
    await page.waitForTimeout(1500)
    await page.screenshot({ path: `${out}-${name}-${theme}.png`, fullPage: true })
  }

  if (errors.length) {
    failures += errors.length
    console.error(`[${theme}] ${errors.length} console errors:`, errors.slice(0, 5))
  } else {
    console.log(`[${theme}] ok`)
  }
  await context.close()
}

await browser.close()
process.exit(failures ? 1 : 0)
