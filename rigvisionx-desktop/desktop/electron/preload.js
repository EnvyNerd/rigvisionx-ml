const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("rigvisionx", {
  version: process.env.npm_package_version || "0.0.0",
  platform: process.platform,
  secrets: {
    getApiKey: () => ipcRenderer.invoke("rigvisionx:key:get"),
    isKeychainAvailable: () => ipcRenderer.invoke("rigvisionx:key:available"),
    setApiKey: (value) => ipcRenderer.invoke("rigvisionx:key:set", value),
    clearApiKey: () => ipcRenderer.invoke("rigvisionx:key:clear"),
  },
});
