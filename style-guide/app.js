// Vanilla SPA interactions for glassmorphism touchscreen UI
(function () {
  // Category state and filtering
  let currentCategory = 'vine'; // 'vine' | 'hazelnut'

  function updateCategoryUI() {
    const isVine = currentCategory === 'vine';
    document.getElementById('catVine').classList.toggle('active', isVine);
    document.getElementById('catVine').setAttribute('aria-selected', String(isVine));
    document.getElementById('catHazel').classList.toggle('active', !isVine);
    document.getElementById('catHazel').setAttribute('aria-selected', String(!isVine));
    const title = isVine ? 'ვაზის დამყნობის მეთოდები (Grapevine Grafting)' : 'თხილის დამყნობის მეთოდები (Hazelnut Grafting)';
    document.getElementById('categoryTitle').textContent = title;

    document.querySelectorAll('.tile').forEach(tile => {
      const tileCat = tile.getAttribute('data-category');
      const isBoth = tileCat === 'both' || tile.hasAttribute('data-role');
      const show = isBoth || tileCat === currentCategory || (!tileCat && isVine);
      tile.style.display = show ? '' : 'none';
    });
  }

  const catVineBtn = document.getElementById('catVine');
  const catHazelBtn = document.getElementById('catHazel');
  if (catVineBtn && catHazelBtn) {
    catVineBtn.addEventListener('click', () => { currentCategory = 'vine'; updateCategoryUI(); });
    catHazelBtn.addEventListener('click', () => { currentCategory = 'hazelnut'; updateCategoryUI(); });
    updateCategoryUI();
  }

  // Handle tile taps to open modal with injected content
  const modal = document.getElementById('modal');
  const modalTitle = document.getElementById('modalTitle');
  const modalSteps = document.getElementById('modalSteps');
  const modalClose = document.getElementById('modalClose');
  const startBtn = document.getElementById('startBtn');

  function openModal(title, steps) {
    modalTitle.textContent = `ინსტრუქცია — ${title}`;
    modalSteps.innerHTML = '';
    steps.forEach(s => {
      const li = document.createElement('li');
      li.textContent = s;
      modalSteps.appendChild(li);
    });
    modal.setAttribute('aria-hidden', 'false');
  }
  function closeModal() {
    modal.setAttribute('aria-hidden', 'true');
  }

  document.querySelectorAll('.tile').forEach(tile => {
    tile.addEventListener('click', () => {
      const method = tile.getAttribute('data-method') || '';
      let steps;
      try { steps = JSON.parse(tile.getAttribute('data-steps') || '[]'); }
      catch { steps = []; }
      openModal(method, steps);
    });
  });

  modalClose.addEventListener('click', closeModal);
  modal.addEventListener('click', (e) => { if (e.target.classList.contains('modal-backdrop')) closeModal(); });
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeModal(); });

  // Placeholder for future navigation to ESP32 live view
  startBtn.addEventListener('click', () => {
    // In future integrate with viewer; for now show subtle feedback.
    startBtn.disabled = true;
    startBtn.textContent = 'მზადება…';
    setTimeout(() => { startBtn.disabled = false; startBtn.textContent = 'დაწყება'; closeModal(); }, 900);
  });
})();


