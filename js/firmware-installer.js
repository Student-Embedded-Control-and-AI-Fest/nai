// NoodleAI Mode 1: show the USB installer only after the factory binary is hosted.
// ESP Web Tools performs actual flashing only after the user clicks its button.
(async function checkFirmwareInstaller() {
  'use strict';
  const status = document.getElementById('firmwareInstallStatus');
  const control = document.getElementById('firmwareInstallControl');
  if (!status || !control) return;
  function message(value, kind) {
    status.textContent = value;
    status.classList.remove('is-ready', 'is-error');
    if (kind) status.classList.add(kind);
  }
  if (!window.isSecureContext) {
    message('USB flashing requires HTTPS or localhost. Publish this website to GitHub Pages.', 'is-error');
    return;
  }
  if (!navigator.serial) {
    message('Web Serial is not available in this browser. Please use desktop Chrome or Edge.', 'is-error');
    return;
  }
  try {
    const manifestUrl = new URL('firmware/manifest.json', document.baseURI);
    const manifestResponse = await fetch(manifestUrl.href, {cache: 'no-store'});
    if (!manifestResponse.ok) throw new Error('Firmware manifest is missing.');
    const manifest = await manifestResponse.json();
    const build = (manifest.builds || []).find(item => item.chipFamily === 'ESP32-S3');
    if (!build || !Array.isArray(build.parts) || build.parts.length !== 1) {
      throw new Error('Expected one merged ESP32-S3 firmware part.');
    }
    const part = build.parts[0];
    if (part.offset !== 0) throw new Error('Factory image must be flashed at offset 0x0000.');
    const binaryUrl = new URL(part.path, manifestUrl);
    if (binaryUrl.origin !== location.origin) throw new Error('Firmware must be hosted on this website.');
    const probe = await fetch(binaryUrl.href, {method: 'HEAD', cache: 'no-store'});
    if (!probe.ok) throw new Error('Firmware binary is not published yet. Ask the project maintainer to upload it.');
    const length = Number(probe.headers.get('content-length'));
    if (Number.isFinite(length) && length > 0 && length < 65536) {
      throw new Error('Published firmware binary appears too small to be valid.');
    }
    control.hidden = false;
    message(`Ready to install ${manifest.name || 'NoodleAI INT8'} · ${manifest.version || 'unversioned'}.`, 'is-ready');
  } catch (err) {
    control.hidden = true;
    message(`Installer not ready: ${err.message}`, 'is-error');
  }
})();
