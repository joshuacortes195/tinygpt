// opens the site in a real chrome, generates a story and clicks through every view
// usage: npm run e2e -- https://joshuacortes195.github.io/tinygpt/
import { chromium } from 'playwright-core'

const base = process.argv[2] || 'http://localhost:5173/'
const problems = []

// uses the chrome that is already installed on this machine
const browser = await chromium.launch({ channel: 'chrome', headless: true, args: ['--enable-unsafe-webgpu'] })

// one pass at a given screen size
async function check(label, viewport) {
  const context = await browser.newContext({ viewport })
  const page = await context.newPage()
  // real errors fail the run, onnx runtime's own warnings are fine
  page.on('console', (m) => {
    if (m.type() === 'error' && !m.text().includes('onnxruntime')) problems.push(`${label}: ${m.text()}`)
  })
  page.on('pageerror', (e) => problems.push(`${label}: ${e.message}`))

  // wait for the model to download and start
  await page.goto(base)
  const generate = page.getByRole('button', { name: 'Generate' })
  await generate.waitFor()
  await page.waitForFunction(
    () => [...document.querySelectorAll('button')].some((b) => b.textContent.includes('Generate') && !b.disabled),
    null,
    { timeout: 180000 },
  )

  // write a story and wait for the speed line that shows up when it is done
  await generate.click()
  const stats = page.locator('section[aria-label="Output"] > p.num')
  await stats.waitFor({ timeout: 300000 })
  console.log(`${label}: ${await page.locator('aside dl').first().innerText().then((t) => t.split('\n')[1])}, ${await stats.innerText()}`)

  // every view has to render something
  await page.getByRole('tab', { name: 'Tokens' }).click()
  if ((await page.getByRole('listitem').count()) === 0) problems.push(`${label}: token view is empty`)
  await page.getByRole('tab', { name: 'Probabilities' }).click()
  await page.getByText('Top guesses for token 1').waitFor({ timeout: 5000 })
  await page.getByRole('tab', { name: 'Attention' }).click()
  await page.getByRole('group', { name: 'Layer' }).waitFor({ timeout: 30000 })

  // the other pages load and nothing scrolls sideways
  for (const hash of ['#/compare', '#/build']) {
    await page.goto(base + hash)
    await page.getByRole('heading', { level: 1 }).first().waitFor()
    const sideways = await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth)
    if (sideways) problems.push(`${label}: ${hash} scrolls sideways`)
  }
  await context.close()
}

try {
  await check('desktop', { width: 1280, height: 900 })
  await check('phone', { width: 390, height: 844 })
} catch (err) {
  problems.push(String(err))
}
await browser.close()

if (problems.length) {
  console.error(problems.join('\n'))
  process.exit(1)
}
console.log('all good')
