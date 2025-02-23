document.addEventListener('DOMContentLoaded', () => {
    const themeToggleBtn = document.querySelector('#theme-toggle');
    themeToggleBtn?.addEventListener('click', function () {
        applyTheme(!document.documentElement.classList.contains('dark'));
    });
    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', (e) => applyTheme(e.matches));
});

function applyTheme(darkMode) {
    if (darkMode) {
        document.documentElement.classList.remove('light');
        document.documentElement.classList.add('dark');
    } else {
        document.documentElement.classList.remove('dark');
        document.documentElement.classList.add('light');
    }
}

const applyDarkMode = window.matchMedia('(prefers-color-scheme: dark)').matches;
applyTheme(applyDarkMode);