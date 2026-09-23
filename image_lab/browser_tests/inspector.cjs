/* Run against manage.py runserver: NODE_PATH=<playwright location> node browser_tests/inspector.cjs */
const { chromium } = require('playwright');
const assert = require('node:assert/strict');
(async () => {
  const browser = await chromium.launch({ headless: true, channel: 'chrome' });
  try {
    const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.goto(process.env.TEST_URL || 'http://127.0.0.1:8765');
    // Generate a real PNG without a checked-in binary fixture.
    const png = await page.evaluate(() => {
      const canvas = document.createElement('canvas');
      canvas.width = 640; canvas.height = 480;
      const ctx = canvas.getContext('2d');
      ctx.fillStyle = '#d5f580'; ctx.fillRect(0, 0, 640, 480);
      ctx.fillStyle = '#191c19';
      for (let x = 0; x < 640; x += 32) ctx.fillRect(x, 0, 16, 480);
      return canvas.toDataURL().split(',')[1];
    });
    await page.locator('#file-input').setInputFiles({ name: 'stripes.png', mimeType: 'image/png', buffer: Buffer.from(png, 'base64') });
    await page.locator('.panel-inspect').first().waitFor();
    const dialog = page.locator('.image-inspector');
    const loaded = () => page.waitForFunction(() => document.querySelector('.inspector-stage img:not([hidden])')?.naturalWidth > 0).catch(async error => { console.log(await dialog.evaluate(el => el.outerHTML)); throw error; });
    const close = async () => { await page.keyboard.press('Escape'); await page.waitForFunction(() => !document.querySelector('dialog').open); };
    for (const op of ['convolve', 'resample', 'noise', 'deblur']) {
      if (op !== 'convolve') {
        await Promise.all([page.waitForResponse(r => r.url().endsWith('/api/process/')), page.locator(`[data-op="${op}"]`).click()]);
      }
      await page.waitForFunction(op => document.querySelector('.panel-download')?.download.startsWith(`image-lab-${op}-`), op);
      await page.locator('#spinner').waitFor({ state: 'hidden' });
      const count = await page.locator('.panel-inspect').count();
      for (let i = 0; i < count; i++) {
        console.log('Checking', op, i);
        const trigger = page.locator('.panel-inspect').nth(i);
        await trigger.focus(); await page.keyboard.press('Enter'); await loaded();
        const info = await page.locator('.inspector-stage img').evaluate(img => ({ w: img.naturalWidth, h: img.naturalHeight, width: img.getBoundingClientRect().width, height: img.getBoundingClientRect().height, src: img.src }));
        assert.equal(info.width, info.w); assert.equal(info.height, info.h);
        assert.equal(info.src, await page.locator('.panel-download').nth(i).evaluate(a => a.href));
        if (op === 'resample' && i > 0 && i < 3) assert.equal(info.w, 160);
        assert.equal(await dialog.locator('.panel-download').getAttribute('download'), await page.locator('.panel-download').nth(i).getAttribute('download'));
        await close();
        assert.equal(await trigger.evaluate(el => el === document.activeElement), true);
      }
    }
    await page.locator('.panel-inspect').first().click(); await loaded();
    for (let i = 0; i < 5; i++) await dialog.getByRole('button', { name: 'Zoom in', exact: true }).click();
    const viewport = page.locator('.inspector-viewport');
    const before = await viewport.evaluate(el => el.scrollLeft);
    const box = await viewport.boundingBox();
    await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
    await page.mouse.down(); await page.mouse.move(box.x + box.width / 2 - 80, box.y + box.height / 2 - 50, { steps: 5 }); await page.mouse.up();
    assert.ok(await viewport.evaluate(el => el.scrollLeft) > before);
    await dialog.getByRole('button', { name: 'Fit to screen' }).click();
    assert.equal(await viewport.evaluate(el => el.scrollWidth <= el.clientWidth && el.scrollHeight <= el.clientHeight), true);
    await dialog.getByRole('button', { name: '1:1 Actual pixels' }).click();
    assert.equal(await dialog.locator('output').textContent(), '100%');
    // Native dialog traps keyboard focus, and X restores page scrolling.
    for (let i = 0; i < 12; i++) { await page.keyboard.press('Tab'); assert.ok(await dialog.evaluate(el => el.contains(document.activeElement))); }
    await dialog.getByRole('button', { name: 'Close image inspector' }).click();
    assert.equal(await page.locator('body').evaluate(el => el.style.overflow), '');
    await page.locator('.panel-inspect').first().click(); await loaded();
    await page.mouse.click(1, 1); await page.waitForFunction(() => !document.querySelector('dialog').open);
    await page.setViewportSize({ width: 390, height: 844 });
    await page.locator('.panel-inspect').first().click(); await loaded();
    await dialog.getByRole('button', { name: 'Fit to screen' }).click();
    assert.ok(await dialog.evaluate(el => el.getBoundingClientRect().right <= innerWidth));
    assert.equal(await viewport.evaluate(el => el.scrollWidth <= el.clientWidth && el.scrollHeight <= el.clientHeight), true);
    await page.screenshot({ path: '/private/tmp/cse220-inspector-mobile.png' });
    await close();
    await page.setViewportSize({ width: 1280, height: 900 });
    await page.locator('.panel-inspect').first().click(); await loaded();
    await page.screenshot({ path: '/private/tmp/cse220-inspector-desktop.png' });
    await close();
    // Failed exports leave close and download usable, with a visible error.
    const source = await page.locator('.panel-download').first().getAttribute('href');
    await page.route(`**${source}`, route => route.abort());
    await page.locator('.panel-inspect').first().click();
    await page.getByText('Could not load the image.', { exact: false }).waitFor();
    assert.equal(await dialog.getByRole('button', { name: 'Zoom in', exact: true }).isDisabled(), true);
    await close();
    assert.deepEqual(errors, []);
    console.log('PASS: all operation panels, export pixels, keyboard/focus, zoom/pan/fit, close paths, mobile, load failure; no browser errors.');
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
