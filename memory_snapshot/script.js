const puppeteer = require('puppeteer');
const fs = require('fs');

(async () => {
  // Launch the browser
  const browser = await puppeteer.launch({ headless: false });

  // Open a new tab
  const page = await browser.newPage();

  // Retrieve URL from command line arguments
  const url = process.argv[2];
  if (!url) {
    console.log('Please provide a URL as a command line argument.');
    process.exit(1);
  }

  // Navigate to the provided URL
  await page.goto(url);

  // Create a new Chrome DevTools Protocol session
  const client = await page.target().createCDPSession();
  await client.send('HeapProfiler.enable');

  // Set an interval to take a heap snapshot every 5 seconds
  setInterval(async () => {
    // Create a new file for each heap snapshot
    const fileName = `heap-${Date.now()}.heapsnapshot`;
    const writeStream = fs.createWriteStream(fileName);

    // When the heap snapshot report progress event is triggered, stop tracking heap objects and end the write stream
    client.once('HeapProfiler.reportHeapSnapshotProgress', async () => {
      await client.send('HeapProfiler.stopTrackingHeapObjects');
      writeStream.end();
    });

    // Start tracking heap objects and take a heap snapshot
    await client.send('HeapProfiler.startTrackingHeapObjects');
    await client.send('HeapProfiler.takeHeapSnapshot', { reportProgress: true });

    // When a heap snapshot chunk is added, write it to the file
    client.on('HeapProfiler.addHeapSnapshotChunk', (params) => {
      writeStream.write(params.chunk);
    });
  }, 5000);
})();
