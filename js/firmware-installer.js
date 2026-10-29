// Dual-board NoodleAI Mode 1 installer: ESP32-S3 Matrix or ESP32 + MPU6050.
// ESP Web Tools chooses a build based on the connected chipFamily.  It does
// not identify the IMU or other board-specific wiring.
(async function checkFirmwareInstaller() {
  'use strict';
  const status = document.getElementById('firmwareInstallStatus');
  const control = document.getElementById('firmwareInstallControl');
  if (!status || !control) return;

  const supported = ['ESP32-S3', 'ESP32'];
  function message(value, kind) {
    status.textContent = value;
    status.classList.remove('is-ready', 'is-error');
    if (kind) status.classList.add(kind);
  }
  control.hidden = true;
  if (!window.isSecureContext) {
    message('USB flashing requires HTTPS or localhost.', 'is-error');
    return;
  }
  if (!navigator.serial) {
    message('USB flashing needs Web Serial. Use desktop Chrome or Edge.', 'is-error');
    return;
  }

  try {
    const manifestUrl = new URL('firmware/manifest.json', document.baseURI);
    const manifestResponse = await fetch(manifestUrl.href, {cache: 'no-store'});
    if (!manifestResponse.ok) throw new Error('Firmware manifest is unavailable.');
    const manifest = await manifestResponse.json();
    const builds = manifest.builds;
    if (!Array.isArray(builds)) throw new Error('Manifest has no firmware builds.');

    // Check both published builds, not just the S3 build. This prevents users
    // selecting a supported chip for which the image has not been uploaded.
    for (const chip of supported) {
      const matches = builds.filter(item => item?.chipFamily === chip);
      if (matches.length !== 1) throw new Error(`Expected exactly one ${chip} firmware build.`);
      const parts = matches[0].parts;
      if (!Array.isArray(parts) || parts.length !== 1 || parts[0].offset !== 0) {
        throw new Error(`${chip} needs one merged factory image at offset 0.`);
      }
      if (typeof parts[0].path !== 'string' || !parts[0].path.endsWith('.bin')) {
        throw new Error(`${chip} factory filename is invalid.`);
      }
      const binaryUrl = new URL(parts[0].path, manifestUrl);
      if (binaryUrl.origin !== location.origin || !binaryUrl.pathname.startsWith(new URL('.', manifestUrl).pathname)) {
        throw new Error(`${chip} firmware must be hosted alongside the manifest.`);
      }
      const probe = await fetch(binaryUrl.href, {method: 'HEAD', cache: 'no-store'});
      if (!probe.ok) throw new Error(`${chip} binary is not published: ${parts[0].path}`);
      const lengthHeader = probe.headers.get('content-length');
      const length = lengthHeader === null ? NaN : Number(lengthHeader);
      if (Number.isFinite(length) && length < 65536) {
        throw new Error(`${chip} binary appears too small to be a factory image.`);
      }
    }
    control.hidden = false;
    message(`Ready for ESP32 and ESP32-S3 · ${manifest.version || 'INT8'}.`, 'is-ready');
  } catch (err) {
    control.hidden = true;
    message(`Installer not ready: ${err.message}`, 'is-error');
  }
})();
