const puppeteer = require("puppeteer");

(async () => {
  // Launch the browser with desired settings
  const browser = await puppeteer.launch({ headless: false, defaultViewport: null });

  try {
    // Get the first tab of the browser
    const [page] = await browser.pages();

    // Define custom functions that will be added to the page. These will log messages when certain events occur.
    // Each function is exposed to the page's JavaScript context, allowing it to be called from within page.evaluateOnNewDocument.
    await page.exposeFunction('_notifyShowDirectoryPicker', (directoryName) => {
      console.log(`showDirectoryPicker was called. Selected directory: ${directoryName}`);
    });

    await page.exposeFunction('_notifyRemoveEntry', (name) => {
      console.log(`removeEntry was called for ${name}.`);
    });

    await page.exposeFunction('_notifyWriteClose', (name) => {
      console.log(`Write.Close was called for ${name}.`);
    });

    await page.exposeFunction('_notifyCreateWritable', (name) => {
      console.log(`CreateWritable was called for ${name}.`);
    });

    await page.exposeFunction('_notifyWrite', (name, content) => {
      console.log(`Write was called for ${name}. Content: ${content}`);
    });

    await page.exposeFunction('_notifyGetFile', (filename) => {
      console.log(`getFile was called for ${filename}.`);
    });

    // Modify the existing File System Access API within the page context to call the custom functions defined above
    // This is done using page.evaluateOnNewDocument, which lets us inject JavaScript that will run as soon as a new page is opened
    await page.evaluateOnNewDocument(() => {
      // Store original FSA API methods
      const originalMethods = {
        showDirectoryPicker: window.showDirectoryPicker.bind(window),
        removeEntry: FileSystemDirectoryHandle.prototype.removeEntry,
        createWritable: FileSystemFileHandle.prototype.createWritable,
        write: FileSystemWritableFileStream.prototype.write,
        close: FileSystemWritableFileStream.prototype.close,
        getFile: FileSystemFileHandle.prototype.getFile,
      };

      // Overwrite FSA API methods with custom ones that also notify us when they're used
      window.showDirectoryPicker = async function() {
        const directoryHandle = await originalMethods.showDirectoryPicker();
        window._notifyShowDirectoryPicker(directoryHandle.name);
        return directoryHandle;
      };

      FileSystemDirectoryHandle.prototype.removeEntry = async function(name, options) {
        window._notifyRemoveEntry(name);
        return originalMethods.removeEntry.call(this, name, options);
      };

      let currentWritableFile = null;
      FileSystemFileHandle.prototype.createWritable = async function(options) {
        const writableStream = await originalMethods.createWritable.call(this, options);
        window._notifyCreateWritable(this.name);
        currentWritableFile = this.name;
        return writableStream;
      };

      FileSystemWritableFileStream.prototype.write = async function(content) {
        window._notifyWrite(currentWritableFile, typeof content === 'object' ? '[object Object]' : content);
        return originalMethods.write.call(this, content);
      };

      FileSystemWritableFileStream.prototype.close = async function() {
        window._notifyWriteClose(currentWritableFile);
        currentWritableFile = null;
        return originalMethods.close.call(this);
      };

      FileSystemFileHandle.prototype.getFile = async function(options) {
        const file = await originalMethods.getFile.call(this, options);
        window._notifyGetFile(this.name);
        return file;
      };
    });

    // Retrieve URL from command line arguments
    const url = process.argv[2]; 
    if (!url) {
      console.log('Please provide a URL as a command line argument.');
      process.exit(1);
    }

    // Navigate to the provided URL
    await page.goto(url);

  } catch (err) {
    // If any error occurs, log it to the console
    console.error(err);
  } finally {
    // Wait for a while before closing the browser to allow asynchronous tasks to complete
    await new Promise(resolve => setTimeout(resolve, 100000));
    await browser.close();
  }
})();
