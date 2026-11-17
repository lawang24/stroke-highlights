export function setupTestClips(onSelect) {
  const button = document.createElement('button');
  button.type = 'button';
  button.textContent = 'Test clips';
  button.setAttribute('aria-haspopup', 'dialog');
  document.getElementById('video-heading').append(button);

  const dialog = document.createElement('dialog');
  dialog.className = 'test-clips-dialog';
  dialog.setAttribute('aria-labelledby', 'test-clips-title');
  dialog.innerHTML = `
    <h2 id="test-clips-title">Test clips</h2>
    <label for="test-clip-select">Local video</label>
    <select id="test-clip-select"></select>
    <p class="hint" role="status"></p>
    <div class="dialog-actions">
      <button type="button" data-action="cancel">Cancel</button>
      <button type="button" data-action="load" class="primary">Load video</button>
    </div>`;
  document.body.append(dialog);
  const select = dialog.querySelector('select');
  const status = dialog.querySelector('[role=status]');
  const load = dialog.querySelector('[data-action=load]');
  let clips = [], controller, selectedId;

  button.addEventListener('click', async () => {
    clips = [];
    select.replaceChildren();
    select.disabled = load.disabled = true;
    status.textContent = 'Loading test clips…';
    dialog.showModal();
    controller = new AbortController();
    try {
      const response = await fetch('/__dev/test-clips', { signal: controller.signal });
      if (!response.ok) throw new Error('Could not list test clips. Try reopening the picker.');
      clips = await response.json();
      for (const group of ['Recordings', 'Regression clips']) {
        const entries = clips.filter(clip => clip.group === group);
        if (!entries.length) continue;
        const optgroup = document.createElement('optgroup');
        optgroup.label = group;
        for (const clip of entries) {
          const option = document.createElement('option');
          option.value = clip.id;
          option.textContent = `${clip.name} · ${(clip.size / 1024 / 1024).toFixed(1)} MB`;
          optgroup.append(option);
        }
        select.append(optgroup);
      }
      if (clips.some(clip => clip.id === selectedId)) select.value = selectedId;
      select.disabled = load.disabled = clips.length === 0;
      status.textContent = clips.length ? 'Loading a clip replaces the current video and shot list.' : 'No local test clips found.';
    } catch (error) {
      if (error.name !== 'AbortError') status.textContent = error.message;
    }
  });
  dialog.addEventListener('close', () => controller?.abort());
  dialog.querySelector('[data-action=cancel]').addEventListener('click', () => dialog.close());
  load.addEventListener('click', () => {
    const clip = clips.find(item => item.id === select.value);
    if (!clip) return;
    selectedId = clip.id;
    dialog.close();
    onSelect(clip);
  });
  return { setBusy(busy) { button.disabled = busy; } };
}
