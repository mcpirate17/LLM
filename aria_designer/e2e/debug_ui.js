import { chromium } from 'playwright';

(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage();
  
  page.on('console', msg => console.log('PAGE LOG:', msg.text()));
  page.on('pageerror', err => console.log('PAGE ERROR:', err.message));

  try {
    console.log('Visiting http://localhost:5176...');
    await page.goto('http://localhost:5176', { waitUntil: 'networkidle' });
    
    const rootContent = await page.innerHTML('#root');
    console.log('Root content length:', rootContent.length);
    if (rootContent.length < 100) {
      console.log('Root content too short, likely crash. Content:', rootContent);
    }
  } catch (err) {
    console.error('Failed to load page:', err);
  } finally {
    await browser.close();
  }
})();
