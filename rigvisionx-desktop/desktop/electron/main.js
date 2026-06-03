const path = require("path");
const { app, BrowserWindow, ipcMain } = require("electron");

const isDev = !!process.env.VITE_DEV_SERVER_URL;
const SERVICE_NAME = "TerraEnergy AI";
const ACCOUNT_NAME = "api_key";
let keytar = null;

try {
  keytar = require("keytar");
} catch {
  keytar = null;
}

function createWindow() {
  const win = new BrowserWindow({
    width: 1280,
    height: 800,
    backgroundColor: "#0f1115",
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  if (isDev) {
    win.loadURL(process.env.VITE_DEV_SERVER_URL);
  } else {
    win.loadFile(path.join(__dirname, "..", "ui", "dist", "index.html"));
  }
}

app.whenReady().then(() => {
  ipcMain.handle("rigvisionx:key:get", async () => {
    if (!keytar) return null;
    return keytar.getPassword(SERVICE_NAME, ACCOUNT_NAME);
  });

  ipcMain.handle("rigvisionx:key:available", async () => {
    return Boolean(keytar);
  });

  ipcMain.handle("rigvisionx:key:set", async (_event, value) => {
    if (!keytar) return false;
    if (!value) {
      await keytar.deletePassword(SERVICE_NAME, ACCOUNT_NAME);
      return true;
    }
    await keytar.setPassword(SERVICE_NAME, ACCOUNT_NAME, String(value));
    return true;
  });

  ipcMain.handle("rigvisionx:key:clear", async () => {
    if (!keytar) return false;
    return keytar.deletePassword(SERVICE_NAME, ACCOUNT_NAME);
  });

  createWindow();

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") {
    app.quit();
  }
});

